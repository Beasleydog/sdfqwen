FROM python:3.12-slim
RUN useradd --create-home --uid 1000 agent && mkdir -p /workspace && chown agent:agent /workspace
USER agent
WORKDIR /workspace
CMD ["sleep", "infinity"]
