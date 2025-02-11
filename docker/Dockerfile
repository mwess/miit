FROM mwess89/greedyfhist:0.0.2-rc2

# RUN pip install --user git+https://github.com/mwess/miit@v0.0.2
RUN pip install git+https://github.com/mwess/miit@v0.0.3

RUN pip install jupyterlab

RUN git clone --depth 1 --branch v0.0.3 https://github.com/mwess/miit

EXPOSE 8888
ENTRYPOINT ["jupyter", "lab"]
CMD ["--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

