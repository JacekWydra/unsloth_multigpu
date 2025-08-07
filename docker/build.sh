#!/bin/bash
TAG=0.1.0

docker build -t nrepo.intra.pixel.com.pl:5000/ds/llm_finetune_causal_reports:$TAG -f docker/Dockerfile .
docker push nrepo.intra.pixel.com.pl:5000/ds/llm_finetune_causal_reports:$TAG