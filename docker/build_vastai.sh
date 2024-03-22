#!/usr/bin/env bash

set +x

TAG=2

docker build . -f Dockerfile_Vastai --tag augunrik/asdfasdfasdfasdf:$TAG

docker push augunrik/asdfasdfasdfasdf:$TAG

