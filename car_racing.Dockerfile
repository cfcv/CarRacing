# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# FROM tensorflow/tensorflow:2.2.0rc1-py3-jupyter
# FROM tensorflow/tensorflow:2.1.0-py3-jupyter
FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN apt-get update && apt-get install -y tmux libsm6 libxext6 libxrender-dev xvfb python-opengl libfontconfig

RUN mkdir /home/code
RUN mkdir /home/share

RUN apt-get install -y git

# Install Atari environment
RUN pip3 install gym[Box2d] tqdm
# RUN pip3 install atari-py
RUN pip3 install --upgrade tensorflow
RUN pip3 install tf-agents==0.4.0rc0
RUN pip3 install pyvirtualdisplay

#RUN git clone https://github.com/google-research/seed_rl.git
# Copy SEED codebase and SEED GRPC binaries.
# ADD . /seed_rl/
# WORKDIR /seed_rl

ENV PYTHONPATH $PYTHONPATH:/
# ENTRYPOINT ["python3", "gcp/run.py"]
COPY main.py /home/code/main.py
COPY SAC.py /home/code/SAC.py
COPY custom_critic_network.py /home/code/custom_critic_network.py
