#add all the local processes to xhost, so the container reaches the window manager
xhost + local:

code_folder="./scripts"
container_name="xrdc_docker"
image_name="xrdc-docker"
image_tag="latest"

# Example
data="/home/barni/Experiments/xr_drone"

#check if correct directory paths are provided
if [ -z "$data" ];
then
    echo "Please provide all directory paths in the ./misc/.params file!"
    exit
fi
if  ! [ -d "$data" ];
then
    echo "Please make sure you provide directories that exist."
    exit
fi

#Set permissions for newly created files. All new files will have the following permissions.
#Important is that the group permissions of the files created are set to read and write and execute.
#We add src as a volume, so we will be able to edit and delete the files created in the container.
setfacl -PRdm u::rwx,g::rwx,o::r ./

#check if container exists
if [[ $( docker ps -a -f name=$container_name | wc -l ) -eq 2 ]];
then
    echo "Container already exists. Do you want to restart it or remove it?"
    select yn in "Restart" "Remove"; do
        case $yn in
            Restart )
                echo "Restarting it... If it was started without USB, it will be restarted without USB.";
                docker restart $container_name;
                break;;
            Remove )
                echo "Stopping it and deleting it... You should simply run this script again to start it.";
                docker stop $container_name;
                docker rm $container_name;
                break;;
        esac
    done
else
    echo "Container does not exist. Creating it."
    #NVIDIA_VISIBLE_DEVICES and NVIDIA_DRIVER_CAPABILITIES sets the visible devices and capabilities of the GPU
    #gpus all adds all the gpus to the container
    #runtime=nvidia tells the docker engine to use the nvidia runtime
    #sometimes GUI programs such as rviz are running faster (OpenGL on NVIDIA) with these params, but you don't need them here
    docker run \
        --env DISPLAY=${DISPLAY} \
        --env NVIDIA_VISIBLE_DEVICES=all \
        --env NVIDIA_DRIVER_CAPABILITIES=all \
        --volume /tmp/.X11-unix:/tmp/.X11-unix \
        --volume ${code_folder}:/home/appuser/xrdc \
        --volume ${data}:/home/appuser/data \
        --network host \
        --interactive \
        --privileged \
        --detach \
        --tty \
        --gpus all \
        --runtime=nvidia \
        --name $container_name \
        --shm-size=8g \
        $image_name:$image_tag 
fi