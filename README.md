# Briefing

This project main goal is to develop my skills in logging, project architecture, ML pipelines, FastAPI and Docker.

# Actual

The project has a consistent pipeline that downloads images and predicts objects by executing the `ml-pipeline` code in the terminal using Bash.

# Development

The image detector it be deployed through a API using FastAPI. Then, it will conteneirized with Docker so that any person can use it.

# Next steps

1. Implement alarm system that warns user if there is no object predicted.
2. Improve post method /detect_upload so it returns the image identified.
3. Ship through docker to test in another computers.