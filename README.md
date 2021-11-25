Test task:<br />
&emsp;Download pandas json : https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json <br />
Context: <br />
&emsp;These are deviations of floor vs ceiling corners of one of our models with ground truth labels for the room name and number of corners in that room with predictions. Please create meaningful statistics of how well the model performed. 

![alt text](https://i.imgur.com/KmX0MW8.png)

Gt_corners = ground truth number of corners in the room <br />
Rb_corners = number of corners found by the model <br />
Mean max min and all others are deviation values in degrees. <br />

1.Create project in idea, pycharm or vscode <br />
2.Create requirements.txt and virtual env <br />
3.Create class for drawing plots <br />
4.Create function “draw_plots” <br />
&emsp;-> reads json file passed as parameter as a pandas dataframe <br />
&emsp;-> draws plot for comparing different columns <br />
&emsp;-> saves plots in a folder called “plots” <br />
&emsp;-> returns paths to all plots <br />
5.Create jupyter notebook called Notebook.ipynb in the root directory to call and visualize our plots <br />

