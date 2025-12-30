from utils.common import read_yaml, read_json_or_jsonl

def load_data(split='', mode=''):
    if split in ["encyclo-k_all", "all_20_perbook_r1_seed_40_5000_full", "all_20_perbook_r1_seed_10_prompt", "all_20_perbook_r1_seed_20_prompt", "all_20_perbook_r1_seed_10_analysis", "all_20_perbook_r1_seed_20_analysis", "all_20_perbook_r1_seed_30_analysis", "all_20_perbook_r1_seed_40_analysis", "all_20_perbook_r1_seed_50_analysis",\
        "all_20_perbook_r1_seed_10", "all_20_perbook_r1_seed_20", "all_20_perbook_r1_seed_30", "all_20_perbook_r1_seed_40", "all_20_perbook_r1_seed_50",\
        "all_20_perbook_r1_seed_40_2000", "all_20_perbook_r1_seed_40_3000", "all_20_perbook_r1_seed_40_4000", "all_20_perbook_r1_seed_40_6000",\
        "all_20_perbook_r1", "all_20_perbook_r1_10", "all_20_perbook_r1_20", "all_20_perbook_r1_30", "all_20_perbook_r1_40", "all_20_perbook_r1_50",\
        "all_20_perbook_r1_para_combination_4", "all_20_perbook_r1_para_combination_5", "all_20_perbook_r1_para_combination_6", "all_20_perbook_r1_para_combination_7", "all_20_perbook_r1_para_combination_8",\
        "all_20_perbook_r1_para_statement_4", "all_20_perbook_r1_para_statement_5", "all_20_perbook_r1_para_statement_6", "all_20_perbook_r1_para_statement_7", "all_20_perbook_r1_para_statement_8", "all_20_perbook_r1_para_statement_9", "all_20_perbook_r1_para_statement_10", "all_20_perbook_r1_para_statement_11", "all_20_perbook_r1_para_statement_12", "all_20_perbook_r1_para_statement_13", "all_20_perbook_r1_para_statement_14", "all_20_perbook_r1_para_statement_15", "all_20_perbook_r1_para_statement_16",\
        "all_20_perbook_r1_para_option_4", "all_20_perbook_r1_para_option_5", "all_20_perbook_r1_para_option_6", "all_20_perbook_r1_para_option_7", "all_20_perbook_r1_para_option_8", "all_20_perbook_r1_para_option_9", "all_20_perbook_r1_para_option_10"] and \
        mode in ['zero-shot', 'zero-shot-bon', 'five-shot']:
        sample = read_json_or_jsonl(f'data', split) # read jsonl in a list
        config = mode.replace('-bon', '')
        template = read_yaml(config)
        for item in sample:
            options_format = ', '.join([chr(65+i) for i in range(len(item['options']))])
            prompt_format = [item['question']+'\n'+'\n'.join([f'{chr(65+i)}) {option}' for i, option in enumerate(item['options'])])]
            prompt = template['prompt_format'][0].format(options_format, *prompt_format)
            yield prompt, item

    elif split == 'SuperGPQA-all' and mode in ['zero-shot-with-subfield']:
        sample = read_json_or_jsonl(f'data', split) # read jsonl in a list
        config = 'zero-shot-with-subfield'
        template = read_yaml(config)
        for item in sample:
            prompt_format = [item['subfield'], item['question']+'\n'+'\n'.join([f'{chr(65+i)}) {option}' for i, option in enumerate(item['options'])])]
            prompt = template['prompt_format'][0].format(*prompt_format)
            yield prompt, item

    elif split == 'SuperGPQA-all' and 'robustness-exp' in mode:
        sample = read_json_or_jsonl(f'data', split) # read jsonl in a list
        config = 'robustness-exp'
        template = read_yaml(config)
        prompt_index, format_index = mode.split('-')[-2], mode.split('-')[-1]

        for item in sample:
            question_format_list = [
                item['question']+ '\n' + '\n'.join([f'{chr(65+i)}) {option}' for i, option in enumerate(item['options'])]),
                item['question']+ '\n' + '\n'.join([f'{chr(65+i)}. {option}' for i, option in enumerate(item['options'])]) + '\n' + 'Your response: ',
                'Question: ' + item['question'] + '\n' + 'Options:\n' + '\n'.join([f'{chr(65+i)}: {option}' for i, option in enumerate(item['options'])]),
                'Question:\n' + item['question'] + '\n' + 'Options:\n' + '\n'.join([f'{chr(65+i)}) {option}' for i, option in enumerate(item['options'])]) + '\n' + 'Please begin answering.',
                'Q: ' + item['question'] + '\n' +' '.join([f'({chr(65+i)}) {option}' for i, option in enumerate(item['options'])]),
                '**Question**:\n' + item['question']+ '\n' + '**Options**:\n' + '\n'.join([f'({chr(65+i)}) {option}' for i, option in enumerate(item['options'])]),            
            ]
            prompt = template[f'initial_prompt_{prompt_index}'][0].format(question_format_list[int(format_index)])
            yield prompt, item
    else:
        raise ValueError(f"Unsupported split '{split}' or mode '{mode}'")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <mode>")
        sys.exit(1)
        
    mode = sys.argv[1]
    last_prompt = None
    from tqdm import tqdm
    for prompt, sample in tqdm(load_data('all_20_perbook_r1_seed_10_analysis', mode), desc='Loading data'):
        last_prompt = prompt
        last_sample = sample
        break

    if last_prompt is not None:
        print(last_prompt)
        print('-'*100)
    # import sys
    # if len(sys.argv) < 2:
    #     print("Usage: python data_loader.py <mode>")
    #     sys.exit(1)
        
    # mode = sys.argv[1]
    # last_prompt = None
    # from tqdm import tqdm
    # for prompt, sample in tqdm(load_data('SuperGPQA-all', mode), desc='Loading data'):
    #     last_prompt = prompt
    #     last_sample = sample
    #     break

    # if last_prompt is not None:
    #     print(last_prompt)
    #     print('-'*100)