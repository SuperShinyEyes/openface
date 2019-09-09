docker:
	docker run --runtime=nvidia --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it --rm -v /l/openface\:/app shinyeyes/openface:v1 /bin/bash