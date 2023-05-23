import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("companys_data.csv")
print(df.columns)

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
df['P/E'] = df['P/E'].replace([pd.NaT, np.inf, -np.inf], 0)

# Calculating P/S column
df['P/S'] = df['capital'] / df['revenue']
df['P/S'] = df['P/S'].replace([np.nan, np.inf, -np.inf], 0)

# Calculating Graham coefficient
df["grahamCoef"] = df["Price"] / ((df["Assets"] - df["Debt"]) / (df["Shares"])) / 10

print(df)

# Filter and print values less than 15 in the "P/E" column
pe_filtered = df[df['P/E'] < 15]
pe_filtered_sorted = pe_filtered.sort_values('P/E', ascending=True)
pe_filtered_values = pe_filtered_sorted[['Company name', 'P/E']]
print("\n\nValues less than 15 in P/E column (sorted in ascending order):")
print(pe_filtered_values.head(30))

# Filter and print values less than 1 in the "P/S" column
ps_filtered = df[df['P/S'] < 1]
ps_filtered_sorted = ps_filtered.sort_values('P/S', ascending=True)
ps_filtered_values = ps_filtered_sorted[['Company name', 'P/S']]
print("\n\nValues less than 1.5 in P/S column (sorted in ascending order):")
print(ps_filtered_values.head(30))

# Sort and print top 30 values from the "ROE" column
roe_sorted = df.sort_values('ROE', ascending=False)
top_30_roe = roe_sorted[['Company name', 'ROE']].head(30)
print("\n\nTop 30 values from ROE column:")
print(top_30_roe)

# Filter and print values from the "grahamCoef" column that are between 40 and 70
graham_filtered = df[(df['grahamCoef'] > 50) & (df['grahamCoef'] < 70)]
graham_filtered_values = graham_filtered[['Company name', 'grahamCoef']].head(30)
print("\n\nValues from grahamCoef column between 50 and 70:")
print(graham_filtered_values)

# Extract top 30 values into sets
pe_set = set(pe_filtered_values['Company name'].head(30))
ps_set = set(ps_filtered_values['Company name'].head(30))
roe_set = set(top_30_roe['Company name'])
graham_set = set(graham_filtered_values['Company name'])

# Find intersections
intersectionn = (pe_set & ps_set) | (pe_set & roe_set) | (pe_set & graham_set) | (ps_set & roe_set) | (
            ps_set & graham_set) | (roe_set & graham_set)

# intersection = (pe_set & ps_set & roe_set) | (pe_set & ps_set & graham_set) | (ps_set & graham_set & roe_set) | (
# pe_set & graham_set & roe_set)
print("\nIntersections:")
for element in intersectionn:
    print(element)

random_values = np.random.choice(list(intersectionn), size=10, replace=False)

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
    yaxis_title='P/S'
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
    yaxis_title='P/E'
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
    yaxis_title='ROE'
)

# Create the figure for grahamCoef
fig_graham = go.Figure()
fig_graham.add_trace(
    go.Bar(
        x=selected_rows['Company name'],
        y=selected_rows['grahamCoef'],
        marker=dict(color=['blue' if (val <= 70 and val >= 50) else 'red' for val in selected_rows['grahamCoef']]),
        opacity=0.6,
        name='grahamCoef'
    )
)
fig_graham.update_layout(
    title='Histogram of Graham Coefficient',
    xaxis_title='Company name',
    yaxis_title='Graham Coefficient'
)

# Display the histograms
fig_ps.show()
fig_pe.show()
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

df['Main Coefficient'] = df['ROECoef'] - df['PSCoef'] - df['PECoef'] - df['Graham Coefficient']
print(df)

main_sorted = df.sort_values('Main Coefficient', ascending=False)
top_30_main = main_sorted[['Company name', 'Main Coefficient']].head(30)
main_set = set(top_30_main['Company name'].head(30))

print("\n\nTop 30 intersection values from Main Coefficient column:")
for elem in main_set.intersection(intersectionn):
    print(elem)

main_sorted = df.sort_values('Main Coefficient', ascending=False)
top_20_main = main_sorted[['Company name', 'Main Coefficient']].head(20)
main_set = set(top_30_main['Company name'].head(20))

print("\n\nTop 20 intersection values from Main Coefficient column:")
for elem in main_set.intersection(intersectionn):
    print(elem)

main_sorted = df.sort_values('Main Coefficient', ascending=False)
top_30_main = main_sorted[['Company name', 'Main Coefficient']].head(10)
main_set = set(top_30_main['Company name'].head(10))

print("\n\nTop 10 intersection values from Main Coefficient column:")
for elem in main_set.intersection(intersectionn):
    print(elem)

print("\n\nTop 30 values from Main Coefficient column:")
print(top_30_main)

df.to_csv('Companies_data_out.csv', index=False)

sample = df[(df['grahamCoef'].abs() < 120) & (df['ROE'] > -100) & (df['P/S'] != 0)]
# Select 20 random samples
random_samples = sample.sample(n=20)

# Plot histogram for 'P/C'
fig_pc = px.histogram(random_samples, x='Company name', title='Histogram of P/C', y='P/S')
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

# Create the figure for ROE
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
        name='grahamCoef (50-70)'
    )
)
fig_grahame.add_trace(
    go.Bar(
        x=random_samples['Company name'][~mask_graham],
        y=random_samples['grahamCoef'][~mask_graham],
        marker=dict(color='red'),
        opacity=0.6,
        name='grahamCoef (not 50-70)'
    )
)

# Update the layout
fig_grahame.update_layout(
    title='Histogram of grahamCoef',
    xaxis_title='Company name',
    yaxis_title='Graham Coefficient'
)

# Display the histograms
'''
fig_pc.show()
fig_ps.show()
fig_pe.show()
fig_pee.show()
fig_roe.show()
fig_roee.show()
fig_graham.show()
fig_grahame.show()
'''

# Тут начинается вторая часть нашего исследования. Принтим графики для 5 необходимых нам компаний
dFive = pd.read_csv("FIVE.csv")
# Create graph 1
Fgraph1 = go.Scatter(
    x=dFive['Datetime'],  # Data from the "Datetime" column
    y=dFive['med'],       # Data from the "med" column
    mode='lines',
    name='med'
)

# Create graph 2
Fgraph2_1 = go.Scatter(
    x=dFive['Datetime'],  # Data from the "Datetime" column
    y=dFive['K'],         # Data from the "K" column
    mode='lines',
    name='K'
)

Fgraph2_2 = go.Scatter(
    x=dFive['Datetime'],  # Data from the "Datetime" column
    y=dFive['D'],         # Data from the "D" column
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
figFiveMid.show()
figFiveKD.show()

# Create graph 1
dKrsb = pd.read_csv("KRSB.csv")
Kgraph1 = go.Scatter(
    x=dKrsb['Datetime'],  # Data from the "Datetime" column
    y=dKrsb['med2'],       # Data from the "med" column
    mode='lines',
    name='med'
)

# Create graph 2
Kgraph2_1 = go.Scatter(
    x=dKrsb['Datetime'],  # Data from the "Datetime" column
    y=dKrsb['K'],         # Data from the "K" column
    mode='lines',
    name='K'
)

Kgraph2_2 = go.Scatter(
    x=dKrsb['Datetime'],  # Data from the "Datetime" column
    y=dKrsb['D2'],         # Data from the "D" column
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
figKrsbMid.show()
figKrsbKD.show()


dMrks = pd.read_csv("MRKS.csv")
Mgraph1 = go.Scatter(
    x=dMrks['Datetime'],  # Data from the "Datetime" column
    y=dMrks['med'],       # Data from the "med" column
    mode='lines',
    name='med'
)

# Create graph 2
Mgraph2_1 = go.Scatter(
    x=dMrks['Datetime'],  # Data from the "Datetime" column
    y=dMrks['K'],         # Data from the "K" column
    mode='lines',
    name='K'
)

Mgraph2_2 = go.Scatter(
    x=dMrks['Datetime'],  # Data from the "Datetime" column
    y=dMrks['D'],         # Data from the "D" column
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
figMrksMid.show()
figMrksKD.show()


dPmsb = pd.read_csv("PMSB.csv")
Pgraph1 = go.Scatter(
    x=dPmsb['Datetime'],  # Data from the "Datetime" column
    y=dPmsb['med'],       # Data from the "med" column
    mode='lines',
    name='med'
)

# Create graph 2
Pgraph2_1 = go.Scatter(
    x=dPmsb['Datetime'],  # Data from the "Datetime" column
    y=dPmsb['K'],         # Data from the "K" column
    mode='lines',
    name='K'
)

Pgraph2_2 = go.Scatter(
    x=dPmsb['Datetime'],  # Data from the "Datetime" column
    y=dPmsb['D'],         # Data from the "D" column
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
figPmsbMid.show()
figPmsbKD.show()



dRkke = pd.read_csv("RKKE.csv")
Rgraph1 = go.Scatter(
    x=dRkke['Datetime'],  # Data from the "Datetime" column
    y=dRkke['med'],       # Data from the "med" column
    mode='lines',
    name='med'
)

# Create graph 2
Rgraph2_1 = go.Scatter(
    x=dRkke['Datetime'],  # Data from the "Datetime" column
    y=dRkke['K'],         # Data from the "K" column
    mode='lines',
    name='K'
)

Rgraph2_2 = go.Scatter(
    x=dRkke['Datetime'],  # Data from the "Datetime" column
    y=dRkke['D'],         # Data from the "D" column
    mode='lines',
    name='D'
)

# Create a subplot with both graphs
figRkkeMid = go.Figure(data=[Rgraph1])

figRkkeKD = go.Figure(data=[Rgraph2_1])
figRkkeKD.add_trace(Rgraph2_2)

# Set the layout for the subplot
figRkkeKD.update_layout(
    title='PMSB',
    xaxis=dict(title='Date & Time'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1, traceorder='normal'),
    width=1280,
    height=720
)

figRkkeMid.update_layout(
    title='PMSB',
    xaxis=dict(title='Date & Time'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1, traceorder='normal'),
    width=1280,
    height=720
)

# Show the plot
figRkkeMid.show()
figRkkeKD.show()