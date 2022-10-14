if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

${cmd} run -v /home/azureuser/Desktop/:/home/azureuser/Desktop/ --shm-size=6gb -it cqrnn:1.0
