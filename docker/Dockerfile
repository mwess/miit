FROM mwess89/greedyfhist:0.0.3-rc6

RUN uv pip install git+https://github.com/mwess/miit@v0.0.3-rc6

RUN uv pip install jupyterlab

RUN git clone --depth 1 --branch v0.0.3-rc6 https://github.com/mwess/greedyfhist
RUN git clone --depth 1 --branch v0.0.3-rc6 https://github.com/mwess/miit

# Install NiftyReg
RUN git clone https://github.com/KCL-BMEIS/niftyreg

WORKDIR "/app/niftyreg"
RUN mkdir build
WORKDIR "/app/niftyreg/build"
RUN cmake ..
RUN make
RUN make install
WORKDIR "/app"
# RUN cd niftyreg && \
#         mkdir build && cd build \
#         cmake .. && \
#         make && \
#         make install

EXPOSE 8888
ENTRYPOINT ["jupyter", "lab"]
CMD ["--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

