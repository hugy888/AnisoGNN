%startup_mtex
path_odf="./textures/";
textures = ["comp","uni","shear","psc"];
for i =1:length(textures)
    tex=textures(i);
    class_name=sprintf("equi_%s", tex);
    path_dream3d="path_to_save"+class_name+"/";

    data = load(path_odf+tex+".mat"); %load odf
    odf=data.odf;
    for j=1:25
        newpath=path_dream3d+sprintf(class_name+"_%02d/", j);
        data_h5read=h5read(newpath+sprintf(class_name+"_%02d.dream3d",j),"/DataContainers/SyntheticVolumeDataContainer/CellEnsembleData/NumFeatures");

        num_grains=data_h5read(2);%number of grains
        ori = odf.discreteSample(num_grains);
        fname = newpath+sprintf(class_name+"_%02d.txt",j);
        export(ori,fname);
    end
end