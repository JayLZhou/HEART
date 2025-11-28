import hashlib
import json
import time
import asyncio
import math
import numpy as np
import pandas as pd

from Option.Config2 import Config
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
from Storage.QueryStorage import QueryStorage
from Storage.NameSpace import Workspace, Namespace
from Storage.OptunaStorage import OptunaStorage

from Provider.LLMProviderRegister import create_llm_instance

from langchain import FewShotPromptTemplate, PromptTemplate

def prepare_configurations(
    hyperparameter_constraints,
    lower_is_better, 
    top_pct, 
    observed_configs, 
    observed_fvals=None, 
    seed=None
):
    '''Prepare and possible (shuffle) the configurations for prompt templates.'''
    examples = []
    
    hyperparameter_names = observed_configs.columns
    observed_configs_ = observed_configs.copy()
    observed_configs = observed_configs_
    
    # shuffle indices to reduce permutation sensitivity
    if seed is not None:
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(observed_configs.index)
        observed_configs = observed_configs.loc[shuffled_indices]
        if observed_fvals is not None:
            observed_fvals = observed_fvals.loc[shuffled_indices]

    # reset index
    observed_configs = observed_configs.reset_index(drop=True)
    if observed_fvals is not None:
        observed_fvals = observed_fvals.reset_index(drop=True)
    
    if observed_fvals is not None:
        if lower_is_better:
            labels = (observed_fvals < np.percentile(observed_fvals, int(top_pct*100))).astype(int)
        else:
            labels = (observed_fvals > np.percentile(observed_fvals, int(100 - top_pct*100))).astype(int)
        
    # serialize the k-shot examples
    for index, row in observed_configs.iterrows():
        row_string = ''
        for i in range(len(row)):
            lower_bound = hyperparameter_constraints[hyperparameter_names[i]][2]
            n_dp = _count_decimal_places(lower_bound) + 2 # number of decimal places
            row_string += f'{hyperparameter_names[i]}: ' + f'{row[i]:.{n_dp}f}' \
                    if isinstance(row[i], float) and not row[i]%1 ==0 else f'{hyperparameter_names[i]}: ' + str(row[i])
            if i != len(row)-1:
                row_string += ', '
        example = {'Q': row_string}
        if observed_fvals is not None:
            row_index = observed_fvals.index.get_loc(index)
            label = f'## {labels.values[row_index][0]} ##'
            example['A'] = label
        examples.append(example)
        
    return examples

def gen_prompt_tempates(
        self,
        task_context, 
        observed_configs, 
        observed_fvals, 
        candidate_configs, 
        n_prompts=1, 
        bootstrapping=False,
        use_context='full_context', 
        use_feature_semantics=True,
        shuffle_features=False,
        apply_warping=False
):
    '''Generate prompt templates for the few-shot learning task.'''

    model = task_context['model']
    task = task_context['task']
    tot_feats = task_context['tot_feats']
    cat_feats = task_context['cat_feats']
    num_feats = task_context['num_feats']
    n_classes = task_context['n_classes']
    n_samples = task_context['num_samples']
    metric = task_context['metric']

    if metric == 'neg_mean_squared_error':
        metric = 'mean squared error'

    if use_context == 'no_context' or not use_feature_semantics:
        metric = 'a metric'
    
    all_prompt_templates = []
    for i in range(n_prompts):
        few_shot_examples = prepare_configurations(task_context['hyperparameter_constraints'], observed_configs, observed_fvals, 
                                                            seed=i, bootstrapping=bootstrapping, use_feature_semantics=use_feature_semantics, 
                                                            shuffle_features=shuffle_features, apply_warping=apply_warping)

        example_template = """
Hyperparameter configuration: {Q}
Performance: {A}"""
        
        example_prompt = PromptTemplate(
            input_variables=["Q", "A"],
            template=example_template
        )

        prefix = ""
        prefix = f"The following are hyperparameter configurations for a {model} and the corresponding performance measured in {metric}."
        if use_context == 'full_context':
            if task == 'classification':
                prefix += f" The model is evaluated on a tabular {task} task and the label contains {n_classes} classes."
            elif task == 'regression':
                prefix += f" The model is evaluated on a tabular {task} task."
            else:
                raise Exception
            prefix += f" The tabular dataset contains {n_samples} samples and {tot_feats} features ({cat_feats} categorical, {num_feats} numerical). "
        prefix += f" Your response should only contain the predicted {metric} in the format ## performance ##."

        suffix = """
Hyperparameter configuration: {Q}
Performance: """

        few_shot_prompt = FewShotPromptTemplate(
            examples=few_shot_examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["Q"],
            example_separator=""
        )
        all_prompt_templates.append(few_shot_prompt)

    query_examples = prepare_configurations(task_context['hyperparameter_constraints'], candidate_configs, 
                                                    seed=None, bootstrapping=False, use_feature_semantics=use_feature_semantics, 
                                                    shuffle_features=shuffle_features, apply_warping=apply_warping)
    return all_prompt_templates, query_examples

class Aquisition:
    def __init__(self, task_context, n_candidates, n_templates, lower_is_better, 
                 jitter=False, rate_limiter=None, warping_transformer=None, chat_engine=None, 
                 prompt_setting=None, shuffle_features=False):
        '''Initialize the LLM Acquisition function.'''
        self.task_context = task_context
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.n_gens = int(n_candidates/n_templates)
        self.lower_is_better = lower_is_better
        self.apply_jitter = jitter
        if warping_transformer is None:
            self.warping_transformer = None
            self.apply_warping = False
        else:
            self.warping_transformer = warping_transformer
            self.apply_warping = True
        self.chat_engine = chat_engine
        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features
        self.llm = create_llm_instance("openai"), 

        assert type(self.shuffle_features) == bool, 'shuffle_features must be a boolean'


    def _jitter(self, desired_fval):
        '''Add jitter to observed fvals to prevent duplicates.'''

        if not self.apply_jitter:
            return desired_fval

        assert hasattr(self, 'observed_best'), 'observed_best must be set before calling _jitter'
        assert hasattr(self, 'observed_worst'), 'observed_worst must be set before calling _jitter'
        assert hasattr(self, 'alpha'), 'alpha must be set before calling _jitter'

        jittered = np.random.uniform(low=min(desired_fval, self.observed_best), 
                                        high=max(desired_fval, self.observed_best), 
                                        size=1).item()

        return jittered


    def _count_decimal_places(self, n):
        '''Count the number of decimal places in a number.'''
        s = format(n, '.10f')
        if '.' not in s:
            return 0
        n_dp = len(s.split('.')[1].rstrip('0'))
        return n_dp

    def _prepare_configurations_acquisition(
        self,
        observed_configs=None, 
        observed_fvals=None, 
        seed=None,
        use_feature_semantics=True,
        shuffle_features=False
    ):
        '''Prepare and (possibly shuffle) few-shot examples for prompt templates.'''
        examples = []
        
        if seed is not None:
            # if seed is provided, shuffle the observed configurations
            np.random.seed(seed)
            shuffled_indices = np.random.permutation(observed_configs.index)
            observed_configs = observed_configs.loc[shuffled_indices]
            if observed_fvals is not None:
                observed_fvals = observed_fvals.loc[shuffled_indices]
        else:
            # if no seed is provided, sort the observed configurations by fvals
            if type(observed_fvals) == pd.DataFrame:
                if self.lower_is_better:
                    observed_fvals = observed_fvals.sort_values(by=observed_fvals.columns[0], ascending=False)
                else:
                    observed_fvals = observed_fvals.sort_values(by=observed_fvals.columns[0], ascending=True)
                observed_configs = observed_configs.loc[observed_fvals.index]

        if shuffle_features:
            # shuffle the columns of observed configurations
            np.random.seed(0)
            shuffled_columns = np.random.permutation(observed_configs.columns)
            observed_configs = observed_configs[shuffled_columns]
            
        # serialize the k-shot examples
        if observed_configs is not None:
            hyperparameter_names = observed_configs.columns
            for index, row in observed_configs.iterrows():
                row_string = '## '
                for i in range(len(row)):
                    hyp_type = self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][0]
                    hyp_transform = self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][1]

                    if use_feature_semantics:
                        row_string += f'{hyperparameter_names[i]}: '
                    else:
                        row_string += f'X{i+1}: '

                    if hyp_type in ['int', 'float']:
                        lower_bound = self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][2][0]
                    else:
                        lower_bound = self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][2][1]
                    n_dp = self._count_decimal_places(lower_bound)
                    value = row[i]
                    if self.apply_warping:
                        if hyp_type == 'int' and hyp_transform != 'log':
                            row_string += str(int(value))
                        elif hyp_type == 'float' or hyp_transform == 'log':
                            row_string += f'{value:.{n_dp}f}'
                        elif hyp_type == 'ordinal':
                            row_string += f'{value:.{n_dp}f}'
                        else:
                            row_string += value

                    else:
                        if hyp_type == 'int':
                            row_string += str(int(value))
                        elif hyp_type in ['float', 'ordinal']:
                            row_string += f'{value:.{n_dp}f}'
                        else:
                            row_string += value

                    if i != len(row)-1:
                        row_string += ', '
                row_string += ' ##'
                example = {'Q': row_string}
                if observed_fvals is not None:
                    row_index = observed_fvals.index.get_loc(index)
                    perf = f'{observed_fvals.values[row_index][0]:.6f}'
                    example['A'] = perf
                examples.append(example)
        elif observed_fvals is not None:
            examples = [{'A': f'{observed_fvals:.6f}'}]
        else:
            raise Exception
            
        return examples
    

    def _gen_prompt_tempates_acquisitions(
        self,
        observed_configs, 
        observed_fvals, 
        desired_fval,
        n_prompts=1,
        use_context='full_context',
        use_feature_semantics=True,
        shuffle_features=False
    ):
        '''Generate prompt templates for acquisition function.'''
        all_prompt_templates = []
        all_query_templates = []

        for i in range(n_prompts):
            few_shot_examples = self._prepare_configurations_acquisition(observed_configs, observed_fvals,seed=i, use_feature_semantics=use_feature_semantics)           # need to update seed?
            jittered_desired_fval = self._jitter(desired_fval)

            # contextual information about the task
            task_context = self.task_context
            model = task_context['model']
            task = task_context['task']
            tot_feats = task_context['tot_feats']
            cat_feats = task_context['cat_feats']
            num_feats = task_context['num_feats']
            n_classes = task_context['n_classes']
            metric = 'mean squared error' if task_context['metric'] == 'neg_mean_squared_error' else task_context['metric']
            num_samples = task_context['num_samples']
            hyperparameter_constraints = task_context['hyperparameter_constraints']
            
            example_template = """
Performance: {A}
Hyperparameter configuration: {Q}"""
            
            example_prompt = PromptTemplate(
                input_variables=["Q", "A"],
                template=example_template
            )

            prefix = f"The following are examples of performance of a {model} measured in {metric} and the corresponding model hyperparameter configurations."
            if use_context == 'full_context':
                if task == 'classification':
                    prefix += f" The model is evaluated on a tabular {task} task containing {n_classes} classes."
                elif task == 'regression':
                    prefix += f" The model is evaluated on a tabular {task} task."
                else:
                    raise Exception
                prefix += f" The tabular dataset contains {num_samples} samples and {tot_feats} features ({cat_feats} categorical, {num_feats} numerical)."
            prefix += f" The allowable ranges for the hyperparameters are:\n"
            for i, (hyperparameter, constraint) in enumerate(hyperparameter_constraints.items()):
                if constraint[0] == 'float':
                    # number of decimal places!!
                    n_dp = self._count_decimal_places(constraint[2][0])
                    if constraint[1] == 'log' and self.apply_warping:
                        lower_bound = np.log10(constraint[2][0])
                        upper_bound = np.log10(constraint[2][1])
                    else:
                        lower_bound = constraint[2][0]
                        upper_bound = constraint[2][1]

                    if use_feature_semantics:
                        prefix += f"- {hyperparameter}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"
                    else:
                        prefix += f"- X{i+1}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"

                    if constraint[1] == 'log' and self.apply_warping:
                        prefix += f" (log scale, precise to {n_dp} decimals)"
                    else:
                        prefix += f" (float, precise to {n_dp} decimals)"
                elif constraint[0] == 'int':
                    if constraint[1] == 'log' and self.apply_warping:
                        lower_bound = np.log10(constraint[2][0])
                        upper_bound = np.log10(constraint[2][1])
                        n_dp = self._count_decimal_places(lower_bound)
                    else:
                        lower_bound = constraint[2][0]
                        upper_bound = constraint[2][1]
                        n_dp = 0

                    if use_feature_semantics:
                        prefix += f"- {hyperparameter}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"
                    else:
                        prefix += f"- X{i+1}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"
                    
                    if constraint[1] == 'log' and self.apply_warping:
                        prefix += f" (log scale, precise to {n_dp} decimals)"
                    else:
                        prefix += f" (int)"

                elif constraint[0] == 'ordinal':
                    if use_feature_semantics:
                        prefix += f"- {hyperparameter}: "
                    else:
                        prefix += f"- X{i+1}: "
                    prefix += f" (ordinal, must take value in {constraint[2]})"

                else:
                    raise Exception('Unknown hyperparameter value type') 

                prefix += "\n"
            prefix += f"Recommend a configuration that can achieve the target performance of {jittered_desired_fval:.6f}. "
            if use_context in ['partial_context', 'full_context']:
                prefix += "Do not recommend values at the minimum or maximum of allowable range, do not recommend rounded values. Recommend values with highest possible precision, as requested by the allowed ranges. "
            prefix += f"Your response must only contain the predicted configuration, in the format ## configuration ##.\n"

            suffix = """
Performance: {A}
Hyperparameter configuration:"""

            few_shot_prompt = FewShotPromptTemplate(
                examples=few_shot_examples,
                example_prompt=example_prompt,
                prefix=prefix,
                suffix=suffix,
                input_variables=["A"],
                example_separator=""
            )
            all_prompt_templates.append(few_shot_prompt)

            query_examples = self._prepare_configurations_acquisition(observed_fvals=jittered_desired_fval, seed=None, shuffle_features=shuffle_features)
            all_query_templates.append(query_examples)

        return all_prompt_templates, all_query_templates
    
    async def _async_generate(self, user_message):
        '''Generate a response from the LLM async.'''
        message = []
        message.append({"role": "system","content": "You are an AI assistant that helps people find information."})
        message.append({"role": "user", "content": user_message})

        MAX_RETRIES = 3

        resp = None
        for retry in range(MAX_RETRIES):
            try:
                start_time = time.time()
                resp = await self.llm.acompletion(
                    messages=message,
                    stream=False,
                    max_tokens=500,
                    format="text"
                )

                break
            except Exception as e:
                print(f'[AF] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...')
                print(resp)
                print(e)

        if resp is None:
            return None

        tot_tokens = resp['usage']['total_tokens']
        tot_cost = 0.0015*(resp['usage']['prompt_tokens']/1000) + 0.002*(resp['usage']['completion_tokens']/1000)

        return resp, tot_cost, tot_tokens


    async def _async_generate_concurrently(self, prompt_templates, query_templates):
        '''Perform concurrent generation of responses from the LLM async.'''

        coroutines = []
        for (prompt_template, query_template) in zip(prompt_templates, query_templates):
            coroutines.append(self._async_generate(prompt_template.format(A=query_template[0]['A'])))

        # coroutines = [self._async_generate(prompt_template.format(A=query_example['A'])) for prompt_template in prompt_templates]
        tasks = [asyncio.create_task(c) for c in coroutines]

        # assert len(tasks) == int(self.n_candidates/self.n_gens)
        assert len(tasks) == int(self.n_templates)

        results = [None]*len(coroutines)

        llm_response = await asyncio.gather(*tasks)

        for idx, response in enumerate(llm_response):
            if response is not None:
                resp, tot_cost, tot_tokens = response
                results[idx] = (resp, tot_cost, tot_tokens)

        return results  # format [(resp, tot_cost, tot_tokens), None, (resp, tot_cost, tot_tokens)]
    
    def _convert_to_json(self, response_str):
        '''Parse LLM response string into JSON.'''
        pairs = response_str.split(',')
        response_json = {}
        for pair in pairs:
            key, value = [x.strip() for x in pair.split(':')]
            response_json[key] = float(value)
            
        return response_json
    
    def _filter_candidate_points(self, observed_points, candidate_points, precision=8):
        '''Filter candidate points that already exist in observed points. Also remove duplicates.'''
        # drop points that already exist in observed points
        rounded_observed = [{key: round(value, precision) for key, value in d.items()} for d in observed_points]
        rounded_candidate = [{key: round(value, precision) for key, value in d.items()} for d in candidate_points]
        filtered_candidates = [x for i, x in enumerate(candidate_points) if rounded_candidate[i] not in rounded_observed]

        def is_within_range(value, allowed_range):
            """Check if a value is within an allowed range."""
            value_type, transform, search_range = allowed_range
            if value_type == 'int':
                [min_val, max_val] = search_range
                if transform == 'log' and self.apply_warping:
                    min_val = np.log10(min_val)
                    max_val = np.log10(max_val)
                    return min_val <= value <= max_val
                else:
                    return min_val <= value <= max_val and int(value) == value
            elif value_type == 'float':                         # THIS MIGHT NEED TO CHANGE, RIGHT NOW IT CAN"T SIT ON THE BOUNDARY
                [min_val, max_val] = search_range
                if transform == 'log' and self.apply_warping:
                    min_val = np.log10(min_val)
                    max_val = np.log10(max_val)
                return min_val <= value <= max_val
            elif value_type == 'ordinal':
                # check that value is in allowed range up to 2 decimal places
                return any(math.isclose(value, x, abs_tol=1e-2) for x in allowed_range[2])
            else:
                raise Exception('Unknown hyperparameter value type')

        def is_dict_within_ranges(d, ranges_dict):
            """Check if all values in a dictionary are within their respective allowable ranges."""
            return all(key in ranges_dict and is_within_range(value, ranges_dict[key]) for key, value in d.items())

        def filter_dicts_by_ranges(dict_list, ranges_dict):
            """Return only those dictionaries where all values are within their respective allowable ranges."""
            return [d for d in dict_list if is_dict_within_ranges(d, ranges_dict)]


        # check that constraints are satisfied
        hyperparameter_constraints = self.task_context['hyperparameter_constraints']
        filtered_candidates = filter_dicts_by_ranges(filtered_candidates, hyperparameter_constraints)

        filtered_candidates = pd.DataFrame(filtered_candidates)
        # drop duplicates
        filtered_candidates = filtered_candidates.drop_duplicates()
        # reset index
        filtered_candidates = filtered_candidates.reset_index(drop=True)
        return filtered_candidates

    
    def get_candidate_points(self, observed_configs, observed_fvals, 
                             use_feature_semantics=True, use_context='full_context', alpha=-0.2):
        '''Generate candidate points for acquisition function.'''
        assert alpha >= -1 and alpha <= 1, 'alpha must be between -1 and 1'
        if alpha == 0:
            alpha = -1e-3 # a little bit of randomness never hurt anyone
        self.alpha = alpha

        start_time = time.time()

        # get desired f_val for candidate points
        range = np.abs(np.max(observed_fvals.values) - np.min(observed_fvals.values))

        if range == 0:
            # sometimes there is no variability in y :')
            range = 0.1*np.abs(np.max(observed_fvals.values))
        alpha_range = [0.1, 1e-2, 1e-3, -1e-3, -1e-2, 1e-1]

        if self.lower_is_better:
            self.observed_best = np.min(observed_fvals.values)
            self.observed_worst = np.max(observed_fvals.values)
            desired_fval = self.observed_best - alpha*range

            while desired_fval <= .00001:  # score can't be negative
                # try first alpha in alpha_range that is lower than current alpha
                for alpha_ in alpha_range:
                    if alpha_ < alpha:
                        alpha = alpha_  # new alpha
                        desired_fval = self.observed_best - alpha*range
                        break
            print(f'Adjusted alpha: {alpha} | [original alpha: {self.alpha}], desired fval: {desired_fval:.6f}')
        else:
            self.observed_best = np.max(observed_fvals.values)
            self.observed_worst = np.min(observed_fvals.values)
            desired_fval = self.observed_best + alpha*range

            while desired_fval >= .9999:  # accuracy can't be greater than 1
                for alpha_ in alpha_range:
                    if alpha_ < alpha:
                        alpha = alpha_  # new alpha
                        desired_fval = self.observed_best + alpha*range
                        break

            print(f'Adjusted alpha: {alpha} | [original alpha: {self.alpha}], desired fval: {desired_fval:.6f}')

        self.desired_fval = desired_fval

        if self.warping_transformer is not None:
            observed_configs = self.warping_transformer.warp(observed_configs)

        prompt_templates, query_templates = self._gen_prompt_tempates_acquisitions(observed_configs, observed_fvals, desired_fval, n_prompts=self.n_templates, use_context=use_context, use_feature_semantics=use_feature_semantics, shuffle_features=self.shuffle_features)

        print('='*100)
        print('EXAMPLE ACQUISITION PROMPT')
        print(f'Length of prompt templates: {len(prompt_templates)}')
        print(f'Length of query templates: {len(query_templates)}')
        print(prompt_templates[0].format(A=query_templates[0][0]['A']))
        print('='*100)

        number_candidate_points = 0
        filtered_candidate_points = pd.DataFrame()

        retry = 0
        while number_candidate_points < 5:
            llm_responses = asyncio.run(self._async_generate_concurrently(prompt_templates, query_templates))

            candidate_points = []
            tot_cost = 0
            tot_tokens = 0
            # loop through n_coroutine async calls
            for response in llm_responses:
                if response is None:
                    continue
                # loop through n_gen responses
                for response_message in response[0]['choices']:
                        response_content = response_message['message']['content']
                        try:
                            response_content = response_content.split('##')[1].strip()
                            candidate_points.append(self._convert_to_json(response_content))
                        except:
                            print(response_content)
                            continue
                tot_cost += response[1]
                tot_tokens += response[2]

            proposed_points = self._filter_candidate_points(observed_configs.to_dict(orient='records'), candidate_points)
            filtered_candidate_points = pd.concat([filtered_candidate_points, proposed_points], ignore_index=True)
            number_candidate_points = filtered_candidate_points.shape[0]

            print(f'Attempt: {retry}, number of proposed candidate points: {len(candidate_points)}, ',
                  f'number of accepted candidate points: {filtered_candidate_points.shape[0]}')


            retry += 1
            if retry > 3:
                print(f'Desired fval: {desired_fval:.6f}')
                print(f'Number of proposed candidate points: {len(candidate_points)}')
                print(f'Number of accepted candidate points: {filtered_candidate_points.shape[0]}')
                if len(candidate_points) > 5:
                    filtered_candidate_points = pd.DataFrame(candidate_points)
                    break
                else:
                    raise Exception('LLM failed to generate candidate points')

        if self.warping_transformer is not None:
            filtered_candidate_points = self.warping_transformer.unwarp(filtered_candidate_points)


        end_time = time.time()
        time_taken = end_time - start_time

        return filtered_candidate_points, tot_cost, time_taken
    

class GenerativeSurrogateModel:
    def __init__(self, task_context, n_gens, lower_is_better, top_pct,
                 n_templates=1, rate_limiter=None, 
                 verbose=False, chat_engine=None):
        '''Initialize the forward LLM surrogate model. This is modelling p(y|x) as in GP/SMAC etc.'''
        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.top_pct = top_pct
        self.n_templates = n_templates
        self.recalibrator = None
        self.chat_engine = chat_engine
        self.verbose = verbose
        self.llm = create_llm_instance("openai"), 

    async def _async_generate(self, few_shot_template, query_example, query_idx):
        '''Generate a response from the LLM async.'''
        prompt = few_shot_template.format(Q=query_example['Q'])

        MAX_RETRIES = 3

        resp = None
        for retry in range(MAX_RETRIES):
            try:
                resp = await self.llm.acompletion(
                    messages=prompt,
                    stream=False,
                    max_tokens=500,
                    format="text"
                )

            except Exception as e:
                print(f'[AF] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...')
                print(resp)
                print(e)

        if resp is None:
            return None

        tot_tokens = resp['usage']['total_tokens']
        tot_cost = 0.0015*(resp['usage']['prompt_tokens']/1000) + 0.002*(resp['usage']['completion_tokens']/1000)

        return query_idx, resp, tot_cost, tot_tokens



    async def _generate_concurrently(self, few_shot_templates, query_examples):
        '''Perform concurrent generation of responses from the LLM async.'''

        coroutines = []
        for template in few_shot_templates:
            for query_idx, query_example in enumerate(query_examples):
                coroutines.append(self._async_generate(template, query_example, query_idx))

        tasks = [asyncio.create_task(c) for c in coroutines]

        results = [[] for _ in range(len(query_examples))]      # nested list

        llm_response = await asyncio.gather(*tasks)

        for response in llm_response:
            if response is not None:
                query_idx, resp, tot_cost, tot_tokens = response
                results[query_idx].append([resp, tot_cost, tot_tokens])

        return results  # format [(resp, tot_cost, tot_tokens), None, (resp, tot_cost, tot_tokens)]

    def process_response(self, all_raw_response):
        all_pred_probs = [] # p(s<\tau | h)
        for raw_response in all_raw_response:
            tokens = raw_response['tokens']
            logprobs = raw_response['top_logprobs']
            pred_index = min((tokens.index(val) for val in ["0", "1"] if val in tokens), default=None)
            if pred_index is None:
                all_pred_probs.append(np.nan)
            else:
                try:
                    prob_1 = logprobs[pred_index]["1"]
                    prob_0 = logprobs[pred_index]["0"]
                    prob_1 = np.exp(prob_1)/(np.exp(prob_1) + np.exp(prob_0))
                    all_pred_probs.append(prob_1)
                except:
                    all_pred_probs.append(np.nan)

        return all_pred_probs

    
    async def _predict(self, all_prompt_templates, query_examples):
        start = time.time()
        all_preds = []
        tot_tokens = 0
        tot_cost = 0

        bool_pred_returned = []

        # make predictions in chunks of 5, for each chunk make concurent calls
        for i in range(0, len(query_examples), 5):
            query_chunk = query_examples[i:i+5]
            chunk_results = await self._generate_concurrently(all_prompt_templates, query_chunk)
            bool_pred_returned.extend([1 if x is not None else 0 for x in chunk_results])                # track effective number of predictions returned

            for _, sample_response in enumerate(chunk_results):
                if not sample_response:     # if sample prediction is an empty list :(
                    sample_preds = [np.nan] * self.n_gens
                else:
                    all_raw_response = [x['logprobs'] for template_response in sample_response for x in template_response[0]['choices'] ]        # fuarr this is some high level programming
                    sample_preds = self.process_response(all_raw_response)
                    tot_cost += sum([x[1] for x in sample_response])
                    tot_tokens += sum([x[2] for x in sample_response])
                all_preds.append(sample_preds)
        
        end = time.time()
        time_taken = end - start

        success_rate = sum(bool_pred_returned)/len(bool_pred_returned)

        pred_probs = np.array(all_preds).astype(float)
        mean_probs = np.nanmean(pred_probs, axis=1)

        return mean_probs, success_rate, tot_cost, tot_tokens, time_taken
    
    async def _evaluate_candidate_points(self, observed_configs, observed_fvals, candidate_configs):
        '''Evaluate candidate points using the LLM model.'''
        all_run_cost = 0
        all_run_time = 0

        all_prompt_templates, query_examples = gen_prompt_tempates(self.task_context, observed_configs, observed_fvals, candidate_configs, 
                                                                   self.lower_is_better, self.top_pct, n_prompts=self.n_templates)
        
        print('*'*100)
        print(f'Number of all_prompt_templates: {len(all_prompt_templates)}')
        print(f'Number of query_examples: {len(query_examples)}')
        print(all_prompt_templates[0].format(Q=query_examples[0]['Q']))
        # print(freeze)


        response = await self._predict(all_prompt_templates, query_examples)

        pred_probs, success_rate, tot_cost, tot_tokens, time_taken = response

        all_run_cost += tot_cost
        all_run_time += time_taken

        return pred_probs, all_run_cost, all_run_time


    def _warp_candidate_points(self, configurations):
        '''Warp candidate points to log scale if necessary.'''
        warped_configs = configurations.copy().to_dict(orient='records')
        hyperparameter_constraints = self.task_context['hyperparameter_constraints']
        for config in warped_configs:
            for hyperparameter, constraint in hyperparameter_constraints.items():
                if constraint[1] == 'log':
                    config[hyperparameter] = np.log10(config[hyperparameter])

        warped_configs = pd.DataFrame(warped_configs)
        return warped_configs
    

    def _unwarp_candidate_points(self, configurations):
        '''Unwarp candidate points from log scale if necessary.'''
        unwarped_configs = configurations.copy().to_dict(orient='records')
        hyperparameter_constraints = self.task_context['hyperparameter_constraints']
        for config in unwarped_configs:
            for hyperparameter, constraint in hyperparameter_constraints.items():
                if constraint[1] == 'log':
                    config[hyperparameter] = 10**config[hyperparameter]

        unwarped_configs = pd.DataFrame(unwarped_configs)
        return unwarped_configs
    

    def select_query_point(self, observed_configs, observed_fvals, candidate_configs, return_raw_preds=False):
        '''Select the next query point using expected improvement.'''
        # warp candidate points
        observed_configs = self._warp_candidate_points(observed_configs)
        candidate_configs = self._warp_candidate_points(candidate_configs)

        pred_probs, cost, time_taken = asyncio.run(self._evaluate_candidate_points(observed_configs, observed_fvals, candidate_configs))

        best_point_index = np.argmax(pred_probs)

        # unwarp candidate points
        candidate_configs = self._unwarp_candidate_points(candidate_configs)

        best_point = candidate_configs.iloc[[best_point_index], :]  # return selected point as dataframe not series

        if return_raw_preds:
            return best_point, pred_probs, cost, time_taken
        else:
            return best_point, cost, time_taken


class LLMBOSampler:
    def __init__(self, config: Config):
        self.config=config
        self.search_space=config.tuner.search_space
        self.workspace = Workspace(self.config.working_dir, self.config.exp_name)
        self.namespace = Namespace(self.workspace)
        self.opt_storage = OptunaStorage(self.namespace)
        self.query_storage = QueryStorage(config, self.namespace)

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial):
        search_space = self.config.tuner.search_space.build_distributions(self.config.tuner.tuner_params)

        def flatten_distributions(d: dict) -> dict:
            """Recursively extract all leaf BaseDistribution objects into a flat dict."""
            flat = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    flat.update(flatten_distributions(v))
                else:
                    flat[k] = v
            return flat

        search_space = flatten_distributions(search_space)
        return search_space

    def sample_relative(self, study: Study, trial, search_space: dict):
        print(study, trial)
        if self._is_first_query():
            random_sample = self.search_space.sample_from_distributions(dists=search_space)
            import pdb
            pdb.set_trace()
            return random_sample
        if self._is_first_trial(study):
            warm_start_point = self._warm_start(study)
            self.query_storage.upsert(study.user_attrs.get("query"))
            return warm_start_point

        acq = Aquisition(
            task_context="",
            n_candidates=10,
            n_templates=1,
            lower_is_better=False
        )
        observed_configs, observed_fvals = self._extract_observations(study)
        candidate_points, cost, time_taken = acq.get_candidate_points(
            observed_configs=observed_configs,
            observed_fvals=observed_fvals,
        )

        surrogate_model = GenerativeSurrogateModel(
            task_context="",
            n_gens=1,
            lower_is_better=False,
            top_pct=0.1,
        )
        best_point, cost, time_taken = surrogate_model.select_query_point(
            observed_configs=observed_configs,
            observed_fvals=observed_fvals,
            candidate_configs=candidate_points
        )

        return best_point.iloc[0].to_dict()

    def before_trial(self, study: Study, trial: FrozenTrial):
        pass

    def _is_first_query(self):
        is_first_query = (self.query_storage.size() == 0)
        import pdb
        pdb.set_trace()
        return is_first_query

    def _is_first_trial(self, study):
        is_first_trial = (len(study.get_trials(deepcopy=False)) == 0)
        import pdb
        pdb.set_trace()
        return is_first_trial

    def _warm_start(self, study, trial):
        nearest_query = self.query_storage.query(study.user_attrs.get("query")["question"], top_k=1)
        nearest_study_name = hashlib.sha256(json.dumps(nearest_query, sort_keys=True).encode()).hexdigest()
        nearest_study = self.opt_storage.load_study(study_name=nearest_study_name, file_path=None)

        complete_trials = [
            t for t in nearest_study.get_trials() if t.state == TrialState.COMPLETE
        ]

        if not complete_trials:
            return None
        
        last_trial = complete_trials[-1]
        return last_trial.params


    
    def sample_independent(self, study: Study, trial: FrozenTrial, name, distribution):
        raise NotImplementedError("I only support relative sampling")

    def _extract_observations(study: Study):
        trials = study.get_trials(deepcopy=False)

        # 过滤成功完成的 trial
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

        # configs → DataFrame
        observed_configs = pd.DataFrame([t.params for t in completed])


        # fvals → Series（支持单目标和多目标）
        if len(completed) > 0 and completed[0].values is not None:
            # 多目标
            observed_fvals = pd.DataFrame([t.values for t in completed])
        else:
            # 单目标
            observed_fvals = pd.Series([t.value for t in completed])

        return observed_configs, observed_fvals