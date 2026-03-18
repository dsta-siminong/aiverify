import pandas as pd 
import json
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px 

with open('../sklearn_report.json') as j:
    data = json.load(j)
with open('../sklearn_report2.json') as j:
    data2 = json.load(j)

with open('boat_classes.json', 'r') as f:
    class_names = json.load(f)

big_df = None; rows = []
for severity in data['GaussianBlur']:
    df = pd.DataFrame.from_dict(data['GaussianBlur'][severity]).T
    df['severity'] = severity
    df['class'] = df.index
    cm_stats = data2['GaussianBlur'][severity]
    for cl,stats in cm_stats.items():
        row = {"severity": severity, "class": cl}
        row.update(stats); rows.append(row)
    if big_df is None:
        big_df = df 
    else:
        big_df = pd.concat([big_df, df])
big_df.to_csv("sklearn_gaussian_blur.csv", index=False)
cm_df = pd.DataFrame(rows)

print(big_df.columns)
print(cm_df.columns)

combined_df = pd.merge(big_df, cm_df, on=['class', 'severity'])
combined_df['preds_population'] = combined_df['TP'] + combined_df['FP']
combined_df['actual_population'] = combined_df['TP'] + combined_df['FN']

combined_df.to_csv("sklearn_gaussian_blur_combined.csv", index=False)

# for i in [str(x) for x in range(16)]:
#     sub_df = combined_df[combined_df['class'] == i].copy()
#     # sub_df = sub_df.sort_values('severity')
#     # sub_df2 = cm_df[cm_df['class'] == i].copy()

#     ax = sub_df.plot(x='severity', y=['precision', 'recall', 'f1-score', 'TP', 'FP', 'FN', 'TN'])
#     ax.figure.set_size_inches(16,9)
#     ax.set_title(f"Sklearn report statistics for Gaussian Blur, class = {class_names[i]}")
#     ax.set_ylim([0,1])
#     ax.figure.savefig(f"sklearn_figure_class_{class_names[i]}.png")
#     print(i)
#     print(sub_df)
#     print()

    # ax2 = sub_df.plot(x='severity', y=['preds_population', 'actual_population'])
    # ax2.figure.set_size_inches(16,9)
    # ax2.set_title(f"Raw class preds/actual statistics for Gaussian Blur, class = {class_names[i]}")
    # ax2.set_ylim([0,1])
    # ax2.figure.savefig(f"sklearn_figure_class_{class_names[i]}.png")

plot_df = combined_df.pivot(index='severity', columns='class', values='preds_population')
plt.figure(figsize=(16,9))

colors = plt.cm.jet(np.linspace(0,1,16))

bottom = None 
for c in [str(x) for x in range(16)]:
    if bottom is None:
        plt.bar(plot_df.index, plot_df[c], label=class_names[c], color=colors[int(c)])
        bottom = plot_df[c].values
    else:
        plt.bar(plot_df.index, plot_df[c], label=class_names[c], color=colors[int(c)], bottom=bottom)
        bottom = bottom + plot_df[c].values 

plt.xlabel('severity'); plt.ylabel('fraction of all samples predicted')
plt.title('predictions per class v augmentation severity');plt.legend(title='class', bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig('all_classes_proportions_barchart.png')


# import plotly.io as pio; pio.renderers.default = 'browser'

long_df = plot_df.reset_index().melt(id_vars='severity', var_name='class', value_name='pred_frac')
long_df = long_df.sort_values(by='class', key=lambda x: x.apply(int))
class_names2 = {str(k):v for k,v in class_names.items()}
long_df["class"] = long_df['class'].map(class_names2)
print(long_df.head())
fx = px.bar(
    long_df, x='severity', y='pred_frac', color='class', title='prediction distribution across classes v severity', 
    color_discrete_sequence=px.colors.sample_colorscale("Jet", [i/15 for i in range(16)])
)
fx.update_layout(barmode='stack', yaxis_title='fraction of all samples predicted', xaxis_title='severity')
fx.write_html("all_classes_proportions_plotly_barchart.html")

# confm_stats = np.array([data2['GaussianBlur'][x] for x in data2['GaussianBlur']])
# tn = confm_stats[:,0]
# fp = confm_stats[:,1]
# fn = confm_stats[:,2]
# fp = confm_stats[:,3]


def visualize_sklearn_results(data, data2, class_names):

    big_df = None; rows = []
    for severity in data['GaussianBlur']:
        df = pd.DataFrame.from_dict(data['GaussianBlur'][severity]).T
        df['severity'] = severity
        df['class'] = df.index
        cm_stats = data2['GaussianBlur'][severity]
        for cl,stats in cm_stats.items():
            row = {"severity": severity, "class": cl}
            row.update(stats); rows.append(row)
        if big_df is None:
            big_df = df 
        else:
            big_df = pd.concat([big_df, df])
    # big_df.to_csv("sklearn_gaussian_blur.csv", index=False)
    cm_df = pd.DataFrame(rows)
    combined_df = pd.merge(big_df, cm_df, on=['class', 'severity'])
    combined_df['preds_population'] = combined_df['TP'] + combined_df['FP']
    combined_df['actual_population'] = combined_df['TP'] + combined_df['FN']

    combined_df.to_csv("sklearn_gaussian_blur_combined.csv", index=False)

    plot_df = combined_df.pivot(index='severity', columns='class', values='preds_population')
    plt.figure(figsize=(16,9))

    colors = plt.cm.jet(np.linspace(0,1,16))

    bottom = None 
    for c in [str(x) for x in range(16)]:
        if bottom is None:
            plt.bar(plot_df.index, plot_df[c], label=class_names[c], color=colors[int(c)])
            bottom = plot_df[c].values
        else:
            plt.bar(plot_df.index, plot_df[c], label=class_names[c], color=colors[int(c)], bottom=bottom)
            bottom = bottom + plot_df[c].values 

    plt.xlabel('severity'); plt.ylabel('fraction of all samples predicted')
    plt.title('predictions per class v augmentation severity')
    plt.legend(title='class', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig('all_classes_proportions_barchart.png')

    long_df = plot_df.reset_index().melt(id_vars='severity', var_name='class', value_name='pred_frac')
    long_df = long_df.sort_values(by='class', key=lambda x: x.apply(int))
    class_names2 = {str(k):v for k,v in class_names.items()}
    long_df["class"] = long_df['class'].map(class_names2)
    print(long_df.head())
    fx = px.bar(
        long_df, x='severity', y='pred_frac', color='class', title='prediction distribution across classes v severity', 
        color_discrete_sequence=px.colors.sample_colorscale("Jet", [i/15 for i in range(16)])
    )
    fx.update_layout(barmode='stack', yaxis_title='fraction of all samples predicted', xaxis_title='severity')
    fx.write_html("all_classes_proportions_plotly_barchart.html")