textures = ["comp","uni","shear","psc"];
for i =1:length(textures)
    tex=textures(i);
    class_name=sprintf("equi_%s", tex);
    path_dream3d="path_to_save/"+class_name+"/";

    for j=1:25
        newpath=path_dream3d+sprintf(class_name+"_%02d/", j);
        grain_id=h5read(newpath+sprintf(class_name+"_%02d.dream3d",j),"/DataContainers/SyntheticVolumeDataContainer/CellData/FeatureIds");
        fname = newpath+sprintf(class_name+"_%02d.mat",j);
        save(fname,'grain_id')

    end
end