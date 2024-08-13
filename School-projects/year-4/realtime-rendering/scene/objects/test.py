import pywavefront


objs = pywavefront.Wavefront("Beach_Ball_v2/13517_Beach_Ball_v2_L3.obj", parse=True)
obj = objs.materials.popitem()[1]
data = obj.vertices

"""objs = pywavefront.Wavefront("lantern/Latern_Rusted_final.obj", parse=True)
obj = objs.materials.popitem()[1]
data = obj.vertices"""

"""objs = pywavefront.Wavefront("95-winner-cup/Winner Cup/WinnerCup.obj", parse=True)
obj = objs.materials.popitem()[1]
data = obj.vertices"""

"""
objs = pywavefront.Wavefront("cube/cube.obj", parse=True)
obj = objs.materials.popitem()[1]
data = obj.vertices
"""

objs = pywavefront.Wavefront("apple/Apple.obj", parse=True)
obj = objs.materials.popitem()[1]
data = obj.vertices
