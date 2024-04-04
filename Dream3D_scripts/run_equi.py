import json
import os
from subprocess import call

textures = ['comp','uni','shear','psc']

for tex in textures: 
    class_name = 'equi_%s' %tex

    mainfolder='path_to_save'
    dream3_doutput='path_to_save/%s/' %class_name
    path_dream3d='dream3d_path'

    if not os.path.exists(dream3_doutput):
            os.makedirs(dream3_doutput)

    for i in range(1,26):
        newpath = dream3_doutput+class_name+'_{:02d}/'.format(i)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        f= open(mainfolder+'equiaxed.json')
        data = json.load(f)

        data['7']['OutputFile'] = newpath+class_name+'_{:02d}.dream3d'.format(i)

        newfile=newpath+class_name+'_{:02d}.json'.format(i)

        with open(newfile, 'w') as f:
            json.dump(data, f)

        runPipe=path_dream3d+'PipelineRunner.exe -p '+newpath+class_name+'_{:02d}.json'.format(i)
        call(runPipe)
