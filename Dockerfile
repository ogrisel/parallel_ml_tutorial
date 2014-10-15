FROM jupyter/demo

USER root
RUN rm -rf ipython_examples *.ipynb
ADD . /home/jupyter
RUN chown jupyter:jupyter . -R
RUN pip3 install psutil

USER jupyter
