ps -ef | grep run_actor | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps -ef | grep run_learner | grep -v grep | awk '{print $2}' | xargs -r kill -9
