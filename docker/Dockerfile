FROM mwess89/greedyfhist:0.0.3-rc3

RUN pip install git+https://github.com/mwess/miit@v0.0.3-rc5

RUN pip install jupyterlab

RUN git clone --depth 1 --branch v0.0.3-rc3 https://github.com/mwess/greedyfhist
RUN git clone --depth 1 --branch v0.0.3-rc5 https://github.com/mwess/miit

# Install NiftyReg
RUN git clone https://github.com/KCL-BMEIS/niftyreg

WORKDIR "/niftyreg"
RUN mkdir build
WORKDIR "/niftyreg/build"
RUN cmake ..
RUN make
RUN make install
WORKDIR "/"
# RUN cd niftyreg && \
#         mkdir build && cd build \
#         cmake .. && \
#         make && \
#         make install

EXPOSE 8888
ENTRYPOINT ["jupyter", "lab"]
CMD ["--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

