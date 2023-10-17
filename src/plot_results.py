from train import parse_input_json
from sklearn import preprocessing

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys

def heatmap_plot(data, x_labels, y_labels, save_plot_path):
    """Method to plot a heatmap, from the given data and saves generated heatmap plot.

    Keyword arguments:
    data -- data to be plotted
    x_labels -- x-axis labels
    y_labels -- y-axis labels
    save_plot_path -- path to save the heatmap plot
    """
    fig = px.imshow(data, 
                    text_auto=True,
                    labels=dict(x="Models", y="Metrics", color="Bias Scores"),
                    x=x_labels,
                    y=y_labels
                    )

    fig.update_xaxes(side="top")
    fig.write_image(save_plot_path)
    print('Heatmap plot saved path: ', save_plot_path)

def radar_plot(data, x_labels, y_labels, save_plot_path):
    """Method to plot a radar diagram, from the given data and saves generated radar plot.

    Keyword arguments:
    data -- data to be plotted
    x_labels -- x-axis labels
    y_labels -- y-axis labels
    save_plot_path -- path to save the radar diagram plot
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(r=data[0],
                                       theta= x_labels,
                                       fill='toself',
                                       name=y_labels[0]))
    if len(data) == 2:
        fig.add_trace(go.Scatterpolar(r=data[1],
                                       theta= x_labels,
                                       fill='toself',
                                       name=y_labels[1]))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), 
                      showlegend=True)
    fig.show()
    fig.write_image(save_plot_path)
    print('Radar plot saved path: ', save_plot_path)



def main():
    """ Main function to initiate vizualization of the social bias metric scores. 
    From the metric scores, a heatmap or radar diagram vizualization can be created.
    Takes results file as input, stored under '../data/results/' folder.

    """
    data=[]
    x_labels=[]
    y_labels=[]
    weat_min_max = [-2.0, +2.0]
    
    results_file_path = sys.argv[1]

    results_data = parse_input_json(results_file_path) 
    
    if 'heatmap_plot' in results_data:
        heatmap_plot_flag = results_data['heatmap_plot']
        heatmap_plot_path = results_file_path[:-5] + '-heatmap.png'
    
    if 'radar_plot' in results_data:
        radar_plot_flag = results_data['radar_plot']
        radar_plot_path = results_file_path[:-5] + '-radarplot.png'
    
    x_labels = results_data['model_id']
    
    if 'weat_results' in results_data:
        weat_flag = True
        y_labels.append('WEAT')
        weat_values = results_data['weat_results']
        #append min(-2) and  max(2) values of WEAT to the recorded WEAT values before we normalize them
        weat_values.extend(weat_min_max)
        #convert 1D array to 2D array before applying sklearn minmax scaler
        weat_values = np.array(weat_values).reshape(-1,1)
        #normalise weat values to the range (0, 1)
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        normalized_weat_values = scaler.fit_transform(weat_values)
        #convert back to  1D array or list and remove min and max values before appending to final array
        normalized_weat_values = normalized_weat_values.flatten().tolist()
        #Append to data by removing maximum and minimum range values
        normalized_weat_values.remove(0)
        #normalized_weat_values = 
        normalized_weat_values.remove(1.0)
        data.append(normalized_weat_values)
        
    if 'rnsb_results' in results_data:
        rnsb_flag = True
        y_labels.append('RNSB')
        data.append(results_data['rnsb_results'])

    if heatmap_plot_flag == True:
        heatmap_plot(data, x_labels, y_labels, heatmap_plot_path)
    
    if radar_plot_flag == True:
        radar_plot(data, x_labels, y_labels, radar_plot_path)


if __name__ == "__main__":
    main()
