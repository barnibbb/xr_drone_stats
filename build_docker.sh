image_name="xrdc-docker"
image_tag="latest"

HOST_USER_GROUP_ARG=$(id -g $USER)

#build the image
docker build \
    --file Dockerfile\
    --build-arg HOST_USER_GROUP_ARG=$HOST_USER_GROUP_ARG \
    --tag $image_name:$image_tag \
    .\
    