#importing pygal
import pygal
# creating a object
dot_chart = pygal.Dot(x_label_rotation=30)
# nameing a title
dot_chart.title = 'A vs B vs C vs D'
# namind different labels
dot_chart.x_labels = ['Richards', 'DeltaBlue', 'Crypto', 'RayTrace', 'EarleyBoyer', 'RegExp', 'Splay', 'NavierStokes']
# adding the data
dot_chart.add('A', [6395, 2212, 7520, 7218, 12464, 1660, 2123, 8607])
dot_chart.add('B', [8473, 9099, 1100, 2651, 6361, 1044, 3797, 9450])
dot_chart.add('C', [3472, 2933, 4503, 5229, 5510, 1828, 9013, 4669])
dot_chart.add('D', [43, 41, 59, 79, 144, 136, 34, 102])
# rendering the file
dot_chart.render_to_file("dot chart.svg")