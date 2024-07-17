FROM mwess89/greedyfhist:0.0.1

RUN pip install --user git+https://github.com/mwess/miit@v0.0.1

RUN pip install jupyterlab

EXPOSE 8888
ENTRYPOINT ["jupyter", "lab"]
CMD ["--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

