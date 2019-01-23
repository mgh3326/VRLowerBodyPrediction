from pymo.parsers import BVHParser
import pymo.viz_tools
import pymo.writers

parser = BVHParser()

parsed_data = parser.parse('./data/test_Take_2019-01-03_03.48.29_PM.bvh')

pymo.viz_tools.print_skel(parsed_data)
writer = pymo.writers.BVHWriter()
f = open("output.bvh", 'w')
writer.write(parsed_data, f)
f.close()
