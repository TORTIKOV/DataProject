import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

KD_printer = False
TOP10_printer = False
DIST_printer = False
Intersections_printer = False
Intersections_printer_2 = False
Main_printer = False

# Import parsed data
df = pd.read_csv("companys_data.csv")
print(df.columns)

# Removing outliers
df = df[df['Company name'] != 'Детский Мир']
df = df[df['Company name'] != 'Белон']
df = df[df['Company name'] != 'ТНС энерго Нижний Новгород']
df = df[df['Company name'] != 'Магнит']
df = df[df['Company name'] != 'Электроцинк']
df = df[df['Company name'] != 'НПО Наука']
df = df[df['Company name'] != 'ЯТЭК']
df = df[df['Company name'] != 'Соллерс']
df = df[df['Company name'] != 'Арсагера']
df = df[df['Company name'] != 'Акрон']
df = df[df['Company name'] != 'Группа Позитив']
df = df[df['Company name'] != 'ЦИАН']

# Calculating P/E column
df['P/E'] = df['Price'] / df['EPS']
df['P/E'] = df['P/E'].replace([np.nan, np.inf, -np.inf], 0)  # Filling space for

# Calculating P/S column
df['P/S'] = df['capital'] / df['revenue']
df['P/S'] = df['P/S'].replace([np.nan, np.inf, -np.inf], 0)  # Filling space for

# Calculating Graham coefficient
df["grahamCoef"] = df["Price"] / ((df["Assets"] - df["Debt"]) / (df["Shares"])) / 10

print(df)

# Filter and print values less than 15 in the "P/E" column
pe_filtered_sorted = df.sort_values('P/E', ascending=True)
pe_filtered_values = pe_filtered_sorted[['Company name', 'P/E']]
print("\n\nValues less than 15 in P/E column:")
print(pe_filtered_values.head(30))
fig_pe = px.histogram(pe_filtered_values.head(10), x='Company name', y='P/E', title='Top 10 P/E values')
fig_pe.update_layout(xaxis_title='Company name', yaxis_title='Value')

# Filter and print values less than 1 in the "P/S" column
ps_filtered_sorted = df.sort_values('P/S', ascending=True)
ps_filtered_values = ps_filtered_sorted[['Company name', 'P/S']]
print("\n\nValues less than 1 in P/S column:")
print(ps_filtered_values.head(30))
fig_ps = px.histogram(ps_filtered_values.head(10), x='Company name', y='P/S', title='Top 10 P/S values')
fig_ps.update_layout(xaxis_title='Company name', yaxis_title='Value')

# Sort and print top 30 values from the "ROE" column
roe_sorted = df.sort_values('ROE', ascending=False)
top_30_roe = roe_sorted[['Company name', 'ROE']].head(30)
print("\n\nTop 30 values from ROE column:")
print(top_30_roe)
fig_roe = px.histogram(top_30_roe.head(10), x='Company name', y='ROE', title='Top 10 ROE values')
fig_roe.update_layout(xaxis_title='Company name', yaxis_title='Value')

# Filter and print values from the "grahamCoef" column that are between 50 and 70
graham_filtered = df[(df['grahamCoef'] > 50) & (df['grahamCoef'] < 70)]
graham_filtered_values = graham_filtered[['Company name', 'grahamCoef']].head(30)
print("\n\nValues from grahamCoef column between 50 and 70:")
print(graham_filtered_values)
fig_graham_coef = px.histogram(graham_filtered_values.sample(10), x='Company name', y='grahamCoef', title='10 Graham Coefficient 50 ≤ values ≤ 70')
fig_graham_coef.update_layout(xaxis_title='Company name', yaxis_title='Value')

# Extract top 30 values into sets
pe_set = set(pe_filtered_values['Company name'].head(30))
ps_set = set(ps_filtered_values['Company name'].head(30))
roe_set = set(top_30_roe['Company name'])
graham_set = set(graham_filtered_values['Company name'])

# Display the histograms
if TOP10_printer:
    fig_ps.show()
    fig_pe.show()
    fig_roe.show()
    fig_graham_coef.show()

# Peeking values from 'P/S', 'P/E', 'ROE', and 'grahamCoef' columns
ps_values = df['P/S']
pe_values = df['P/E']
roe_values = df['ROE']
graham_coef_values = df['grahamCoef']

# Setting the step size for the histogram (change as desired)
stepPS = 0.05
stepPE = 0.5
stepROE = 5
stepGR = 5

# Setting the range and step size for the histogram (change as desired)
range_ps = (-3, 2)  # Example range for P/S column
range_pe = (-11, 23)    # Example range for P/E column
range_roe = (-50, 120) # Example range for ROE column
range_graham_coef = (0, 100)  # Example range for grahamCoef column
fig_ps = go.Figure()
fig_ps.add_trace(go.Histogram(x=ps_values, xbins=dict(start=range_ps[0], end=range_ps[1], size=stepPS), name='Companies'))
fig_ps.update_layout(title='P/S Values Histogram', xaxis_title='Value', yaxis_title='Quantity')

fig_pe = go.Figure()
fig_pe.add_trace(go.Histogram(x=pe_values, xbins=dict(start=range_pe[0], end=range_pe[1], size=stepPE), name='Companies'))
fig_pe.update_layout(title='P/E Values Histogram', xaxis_title='Value', yaxis_title='Quantity')

fig_roe = go.Figure()
fig_roe.add_trace(go.Histogram(x=roe_values, xbins=dict(start=range_roe[0], end=range_roe[1], size=stepROE), name='Companies'))
fig_roe.update_layout(title='ROE Values Histogram', xaxis_title='Value', yaxis_title='Quantity')

fig_graham_coef = go.Figure()
fig_graham_coef.add_trace(go.Histogram(x=graham_coef_values, xbins=dict(start=range_graham_coef[0], end=range_graham_coef[1], size=stepGR), histnorm='percent', name='Companies'))
fig_graham_coef.update_layout(title='Graham Coefficient Values Histogram', xaxis_title='Value', yaxis_title='Quantity')

# Displaying the line graphs
if DIST_printer:
    fig_ps.show()
    fig_pe.show()
    fig_roe.show()
    fig_graham_coef.show()

sample = df[(df['grahamCoef'].abs() < 120) & (df['ROE'] > -100) & (df['P/S'] != 0)]
# Select 20 random samples
random_samples = sample.sample(n=20)

# Plot histogram for 'P/C'
fig_pc = px.histogram(random_samples, x='Company name', title='Histogram of P/S', y='P/S')
fig_pc.update_xaxes(title='Companies name')
fig_pc.update_yaxes(title='P/S')

# Create a mask to identify values under 1
mask = random_samples['P/S'] < 1

# Create the figure
fig_ps = go.Figure()

# Add bars for values under 1
fig_ps.add_trace(
    go.Bar(
        x=random_samples['Company name'][mask],
        y=random_samples['P/S'][mask],
        marker=dict(color='blue'),
        opacity=0.6,
        name='P/S < 1'
    )
)

# Add bars for values 1 and above
fig_ps.add_trace(
    go.Bar(
        x=random_samples['Company name'][~mask],
        y=random_samples['P/S'][~mask],
        marker=dict(color='red'),
        opacity=0.6,
        name='P/S >= 1'
    )
)

# Update the layout
fig_ps.update_layout(
    title='Histogram of P/S',
    xaxis_title='Company name',
    yaxis_title='P/S'
)

# Plot histogram for 'P/E'
fig_pe = px.histogram(random_samples, x='Company name', title='Histogram of P/E', y='P/E')
fig_pe.update_xaxes(title='Companies name')
fig_pe.update_yaxes(title='P/E')

# Plot histogram for 'ROE'
fig_roe = px.histogram(random_samples, x='Company name', title='Histogram of ROE', y='ROE')
fig_roe.update_xaxes(title='Companies name')
fig_roe.update_yaxes(title='ROE')

# Plot histogram for 'grahamCoef'
fig_graham = px.histogram(random_samples, x='Company name', title='Histogram of Graham Coefficient', y='grahamCoef')
fig_graham.update_xaxes(title='Companies name')
fig_graham.update_yaxes(title='Graham Coefficient')

# Create masks to identify values under 1
mask_pe = random_samples['P/E'] < 15

# Create the figure for P/E
fig_pee = go.Figure()
fig_pee.add_trace(
    go.Bar(
        x=random_samples['Company name'][mask_pe],
        y=random_samples['P/E'][mask_pe],
        marker=dict(color='blue'),
        opacity=0.6,
        name='P/E < 15'
    )
)
fig_pee.add_trace(
    go.Bar(
        x=random_samples['Company name'][~mask_pe],
        y=random_samples['P/E'][~mask_pe],
        marker=dict(color='red'),
        opacity=0.6,
        name='P/E >= 15'
    )
)

fig_pee.update_layout(
    title='Histogram of P/E',
    xaxis_title='Company name',
    yaxis_title='P/E'
)

# Define the colors for the gradient (blue to red)
colorscale = [[0, 'red'], [1, 'blue']]

# Create the figure for ROE
fig_roee = go.Figure()

# Add bars with the gradient color
fig_roee.add_trace(
    go.Bar(
        x=random_samples['Company name'],
        y=random_samples['ROE'],
        marker=dict(
            color=random_samples['ROE'],
            colorscale=colorscale,
            showscale=True
        ),
        opacity=0.6,
        name='ROE'
    )
)

# Update the layout
fig_roee.update_layout(
    title='Histogram of ROE',
    xaxis_title='Company name',
    yaxis_title='ROE'
)

# Create a mask to identify values within the range [50, 70]
mask_graham = (random_samples['grahamCoef'] >= 50) & (random_samples['grahamCoef'] <= 70)

# Create the figure for grahamCoef
fig_grahame = go.Figure()

# Add bars with the specified colors based on the mask
fig_grahame.add_trace(
    go.Bar(
        x=random_samples['Company name'][mask_graham],
        y=random_samples['grahamCoef'][mask_graham],
        marker=dict(color='blue'),
        opacity=0.6,
        name='Graham (50-70)'
    )
)
fig_grahame.add_trace(
    go.Bar(
        x=random_samples['Company name'][~mask_graham],
        y=random_samples['grahamCoef'][~mask_graham],
        marker=dict(color='red'),
        opacity=0.6,
        name='Graham (not 50-70)'
    )
)

# Update the layout
fig_grahame.update_layout(
    title='Histogram of Graham Coefficient',
    xaxis_title='Company name',
    yaxis_title='Graham Coefficient'
)

# Display the histograms of random
if Intersections_printer_2:
    fig_pc.show()
    fig_ps.show()
    fig_pe.show()
    fig_pee.show()
    fig_roe.show()
    fig_roee.show()
    fig_graham.show()
    fig_grahame.show()

# Find intersections in sets
intersections = (pe_set & ps_set) | (pe_set & roe_set) | (pe_set & graham_set) | (ps_set & roe_set) | (
        ps_set & graham_set) | (roe_set & graham_set)

# intersection = (pe_set & ps_set & roe_set) | (pe_set & ps_set & graham_set) | (ps_set & graham_set & roe_set) | (
# pe_set & graham_set & roe_set) That was an attempt to see the intersections of three.
print("\nIntersections:")
for element in intersections:
    print(element)

random_values = np.random.choice(list(intersections), size=10, replace=False)

# Find the selected values in the 'Company name' column
selected_rows = df[df['Company name'].isin(random_values)]

# Create the figure for P/S
fig_ps = go.Figure()
fig_ps.add_trace(
    go.Bar(
        x=selected_rows['Company name'],
        y=selected_rows['P/S'],
        marker=dict(color=['blue' if val < 1 else 'red' for val in selected_rows['P/S']]),
        opacity=0.6,
        name='P/S'
    )
)
fig_ps.update_layout(
    title='Histogram of P/S',
    xaxis_title='Company name',
    yaxis_title='Value'
)

# Create the figure for P/E
fig_pe = go.Figure()
fig_pe.add_trace(
    go.Bar(
        x=selected_rows['Company name'],
        y=selected_rows['P/E'],
        marker=dict(color=['blue' if val < 15 else 'red' for val in selected_rows['P/E']]),
        opacity=0.6,
        name='P/E'
    )
)
fig_pe.update_layout(
    title='Histogram of P/E',
    xaxis_title='Company name',
    yaxis_title='Value'
)

# Create the figure for ROE
colorscale = [[0, 'red'], [1, 'blue']]
fig_roe = go.Figure()
fig_roe.add_trace(
    go.Bar(
        x=selected_rows['Company name'],
        y=selected_rows['ROE'],
        marker=dict(color=selected_rows['ROE'], colorscale=colorscale, showscale=True),
        opacity=0.6,
        name='ROE'
    )
)
fig_roe.update_layout(
    title='Histogram of ROE',
    xaxis_title='Company name',
    yaxis_title='Value'
)

# Create the figure for grahamCoef
fig_graham = go.Figure()
fig_graham.add_trace(
    go.Bar(
        x=selected_rows['Company name'],
        y=selected_rows['grahamCoef'],
        marker=dict(color=['blue' if (70 >= val >= 50) else 'red' for val in selected_rows['grahamCoef']]),
        opacity=0.6,
        name='Graham Coefficient'
    )
)
fig_graham.update_layout(
    title='Histogram of Graham Coefficient',
    xaxis_title='Company name',
    yaxis_title='Value'
)

# Display the histograms
if Intersections_printer:
    fig_pe.show()
    fig_ps.show()
    fig_roe.show()
    fig_graham.show()


# Get min and max values from 'P/S', 'P/E', and 'ROE' columns
ps_min = df['P/S'].min()
ps_max = df['P/S'].max()

pe_min = df['P/E'].min()
pe_max = df['P/E'].max()

roe_min = df['ROE'].min()
roe_max = df['ROE'].max()

# Calculate coefficient values and add new columns
df['PSCoef'] = ((df['P/S'] - ps_min) / (ps_max - ps_min)) * 100
df['PECoef'] = ((df['P/E'] - pe_min) / (pe_max - pe_min)) * 100
df['ROECoef'] = ((df['ROE'] - roe_min) / (roe_max - roe_min)) * 100

# Print the updated DataFrame
print(df)

# Step 1: Calculate 'Graham Difference' column
df['Graham Difference'] = df['grahamCoef'].apply(lambda x: min(abs(x - 50), abs(x - 70)))

# Step 2: Calculate 'Graham Coefficient' column
graham_diff_min = df['Graham Difference'].min()
graham_diff_max = df['Graham Difference'].max()
df['Graham Coefficient'] = ((df['Graham Difference'] - graham_diff_min) / (graham_diff_max - graham_diff_min)) * 100

# Calculating our coefficient
df['Main Coefficient'] = df['ROECoef'] - df['PSCoef'] - df['PECoef'] - df['Graham Coefficient']

selected_rows = df[df['Company name'].isin(random_values)]

fig_main = go.Figure()
fig_main.add_trace(
    go.Bar(
        x=selected_rows['Company name'],
        y=selected_rows['Main Coefficient'],
        marker=dict(color=selected_rows['Main Coefficient'], colorscale=colorscale, showscale=True),
        opacity=0.6,
        name='Main Coefficient'
    )
)
fig_main.update_layout(
    title='Histogram of Main Coefficient',
    xaxis_title='Company name',
    yaxis_title='Value'
)

if Main_printer:
    fig_main.show()

main_sorted = df.sort_values('Main Coefficient', ascending=False)
top_30_main = main_sorted[['Company name', 'Main Coefficient']].head(30)
top_10_main = main_sorted[['Company name', 'Main Coefficient']].head(10)

fig_main = go.Figure()
fig_main.add_trace(
    go.Bar(
        x=top_10_main['Company name'],
        y=top_10_main['Main Coefficient'],
        marker=dict(color=top_10_main['Main Coefficient'], colorscale=colorscale, showscale=True),
        opacity=0.6,
        name='Main Coefficient'
    )
)
fig_main.update_layout(
    title='Top 10 Main Coefficient values',
    xaxis_title='Company name',
    yaxis_title='Value'
)

if Main_printer:
    fig_main.show()

main_set = set(top_30_main['Company name'])

print("\n\nTop 30 intersection values from Main Coefficient column:")
for elem in main_set.intersection(intersections):
    print(elem)

print("\n\nTop 30 values from Main Coefficient column:")
print(top_30_main)

df.to_csv('Companies_data_out.csv', index=False)

# Тут начинается вторая часть нашего исследования. Принтим графики для 5 необходимых нам компаний
dFive = pd.read_csv("FIVE.csv")
# Create graph 1
Fgraph1 = go.Scatter(
    x=dFive['Datetime'],  # Data from the "Datetime" column
    y=dFive['med'],  # Data from the "med" column
    mode='lines',
    name='med'
)

# Create graph 2
Fgraph2_1 = go.Scatter(
    x=dFive['Datetime'],  # Data from the "Datetime" column
    y=dFive['K'],  # Data from the "K" column
    mode='lines',
    name='K'
)

Fgraph2_2 = go.Scatter(
    x=dFive['Datetime'],  # Data from the "Datetime" column
    y=dFive['D'],  # Data from the "D" column
    mode='lines',
    name='D'
)

# Create a subplot with both graphs
figFiveMid = go.Figure(data=[Fgraph1])

figFiveKD = go.Figure(data=[Fgraph2_1])
figFiveKD.add_trace(Fgraph2_2)

# Set the layout for the subplot
figFiveKD.update_layout(
    title='FIVE',
    xaxis=dict(title='Date & Time'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1, traceorder='normal'),
    width=1280,
    height=720
)

figFiveMid.update_layout(
    title='FIVE',
    xaxis=dict(title='Date & Time'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1, traceorder='normal'),
    width=1280,
    height=720
)

# Show the plot
if KD_printer:
    figFiveMid.show()
    figFiveKD.show()

# Create graph 1
dKrsb = pd.read_csv("KRSB.csv")
Kgraph1 = go.Scatter(
    x=dKrsb['Datetime'],  # Data from the "Datetime" column
    y=dKrsb['med2'],  # Data from the "med" column
    mode='lines',
    name='med'
)

# Create graph 2
Kgraph2_1 = go.Scatter(
    x=dKrsb['Datetime'],  # Data from the "Datetime" column
    y=dKrsb['K'],  # Data from the "K" column
    mode='lines',
    name='K'
)

Kgraph2_2 = go.Scatter(
    x=dKrsb['Datetime'],  # Data from the "Datetime" column
    y=dKrsb['D2'],  # Data from the "D" column
    mode='lines',
    name='D'
)

# Create a subplot with both graphs
figKrsbMid = go.Figure(data=[Kgraph1])

figKrsbKD = go.Figure(data=[Kgraph2_1])
figKrsbKD.add_trace(Kgraph2_2)

# Set the layout for the subplot
figKrsbKD.update_layout(
    title='KRSB',
    xaxis=dict(title='Date & Time'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1, traceorder='normal'),
    width=1280,
    height=720
)

figKrsbMid.update_layout(
    title='KRSB',
    xaxis=dict(title='Date & Time'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1, traceorder='normal'),
    width=1280,
    height=720
)

# Show the plot
if KD_printer:
    figKrsbMid.show()
    figKrsbKD.show()

dMrks = pd.read_csv("MRKS.csv")
Mgraph1 = go.Scatter(
    x=dMrks['Datetime'],  # Data from the "Datetime" column
    y=dMrks['med'],  # Data from the "med" column
    mode='lines',
    name='med'
)

# Create graph 2
Mgraph2_1 = go.Scatter(
    x=dMrks['Datetime'],  # Data from the "Datetime" column
    y=dMrks['K'],  # Data from the "K" column
    mode='lines',
    name='K'
)

Mgraph2_2 = go.Scatter(
    x=dMrks['Datetime'],  # Data from the "Datetime" column
    y=dMrks['D'],  # Data from the "D" column
    mode='lines',
    name='D'
)

# Create a subplot with both graphs
figMrksMid = go.Figure(data=[Mgraph1])

figMrksKD = go.Figure(data=[Mgraph2_1])
figMrksKD.add_trace(Mgraph2_2)

# Set the layout for the subplot
figMrksKD.update_layout(
    title='MRKS',
    xaxis=dict(title='Date & Time'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1, traceorder='normal'),
    width=1280,
    height=720
)

figMrksMid.update_layout(
    title='MRKS',
    xaxis=dict(title='Date & Time'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1, traceorder='normal'),
    width=1280,
    height=720
)

# Show the plot
if KD_printer:
    figMrksMid.show()
    figMrksKD.show()

dPmsb = pd.read_csv("PMSB.csv")
Pgraph1 = go.Scatter(
    x=dPmsb['Datetime'],  # Data from the "Datetime" column
    y=dPmsb['med'],  # Data from the "med" column
    mode='lines',
    name='med'
)

# Create graph 2
Pgraph2_1 = go.Scatter(
    x=dPmsb['Datetime'],  # Data from the "Datetime" column
    y=dPmsb['K'],  # Data from the "K" column
    mode='lines',
    name='K'
)

Pgraph2_2 = go.Scatter(
    x=dPmsb['Datetime'],  # Data from the "Datetime" column
    y=dPmsb['D'],  # Data from the "D" column
    mode='lines',
    name='D'
)

# Create a subplot with both graphs
figPmsbMid = go.Figure(data=[Pgraph1])

figPmsbKD = go.Figure(data=[Pgraph2_1])
figPmsbKD.add_trace(Pgraph2_2)

# Set the layout for the subplot
figPmsbKD.update_layout(
    title='PMSB',
    xaxis=dict(title='Date & Time'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1, traceorder='normal'),
    width=1280,
    height=720
)

figPmsbMid.update_layout(
    title='PMSB',
    xaxis=dict(title='Date & Time'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1, traceorder='normal'),
    width=1280,
    height=720
)

# Show the plot
if KD_printer:
    figPmsbMid.show()
    figPmsbKD.show()

dRkke = pd.read_csv("RKKE.csv")
Rgraph1 = go.Scatter(
    x=dRkke['Datetime'],  # Data from the "Datetime" column
    y=dRkke['med'],  # Data from the "med" column
    mode='lines',
    name='med'
)

# Create graph 2
Rgraph2_1 = go.Scatter(
    x=dRkke['Datetime'],  # Data from the "Datetime" column
    y=dRkke['K'],  # Data from the "K" column
    mode='lines',
    name='K'
)

Rgraph2_2 = go.Scatter(
    x=dRkke['Datetime'],  # Data from the "Datetime" column
    y=dRkke['D'],  # Data from the "D" column
    mode='lines',
    name='D'
)

# Create a subplot with both graphs
figRkkeMid = go.Figure(data=[Rgraph1])

figRkkeKD = go.Figure(data=[Rgraph2_1])
figRkkeKD.add_trace(Rgraph2_2)

# Set the layout for the subplot
figRkkeKD.update_layout(
    title='RKKE',
    xaxis=dict(title='Date & Time'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1, traceorder='normal'),
    width=1280,
    height=720
)

figRkkeMid.update_layout(
    title='RKKE',
    xaxis=dict(title='Date & Time'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1, traceorder='normal'),
    width=1280,
    height=720
)

# Show the plot
if KD_printer:
    figRkkeMid.show()
    figRkkeKD.show()
