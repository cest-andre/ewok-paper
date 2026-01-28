import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem


#   TODO:  parse through the ewok results csv files.  For each file, get the four log prob scores.
#          simply compute if logp_target1_context1 > logp_target2_context1 AND logp_target2_context2 > logp_target1_context2
#          If true, score is 1, else 0.  Take average over all dataset versions.
basedir = '/home/alongon/code/ewok-paper/output/results/ewok-core-1.0'
domains = [
    'agent_properties', 'material_dynamics', 'material_properties', 'physical_dynamics', 'physical_interactions', 'physical_relations',
    'quantitative_properties', 'social_interactions', 'social_properties', 'social_relations', 'spatial_relations'
]
# base_model = 'gemma_2_9b_it' 'Mistral_7B_Instruct_v0.2' 'Llama_3.1_8B_Instruct'
# models = [base_model, f'VLM_{base_model}']
models = ['gemma_3_270m_it', 'gemma_3_1b_it']

all_results = {}
for domain in domains:
    all_results[domain] = {}
    for model in models:
        all_results[domain][model] = []
        for data_vers in os.listdir(basedir):
            results = []
            with open(os.path.join(basedir, data_vers, 'eval=logprobs', f'model={model}', f'results-{domain}.csv'), newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    keys = ['MetaTemplateID'] + row[None]
                    break
                reader = csv.DictReader(csvfile, fieldnames=keys)
                for row in reader:
                    score = (float(float(row['logp_target1_context1']) > float(row['logp_target1_context2'])) + float(float(row['logp_target2_context2']) > float(row['logp_target2_context1']))) / 2
                    results.append(score)

            all_results[domain][model].append(np.mean(np.array(results)))

fig, ax = plt.subplots(layout='constrained')
width = 0.25
bars = [[], []]
for i in range(len(models)):
    model = models[i]
    color = 'c' if i == 0 else 'm'
    for j in range(len(domains)):
        label = model if j == 0 else None
        domain = domains[j]
        scores = np.array(all_results[domain][model])

        bar = ax.bar(j+(i*width), np.mean(scores), width, yerr=sem(scores), color=color, alpha=0.3, label=label)
        bars[i].append(bar)
        ax.scatter(np.full((scores.shape), j+(i*width)) + np.random.uniform(low=-0.1, high=0.1, size=scores.shape), scores, s=5, color=color)

        if i == 1:
            ratio = (((np.mean(np.array(all_results[domain][models[0]]))) / np.mean(scores)) - 1) * 100
            ax.bar_label(bars[ratio < 0][j], labels=['{:+0.2f}%'.format(ratio)], padding=3)

ax.set_ylabel('Accuracy (LogProb)')
ax.set_xticks(np.arange(len(domains)) + (0.5*width), [s.replace('_', '\n') for s in domains], size=8, rotation=45)
ax.legend(loc='upper left', ncols=1)
ax.set_ylim(0.5, 1)
ax.set_title('Gemma3 270m vs. 1B EWoK')

plt.savefig('/home/alongon/figures/ewok_vlm/gemma3_270m_v_1b_logprob_scatter.png', bbox_inches='tight', dpi=300)