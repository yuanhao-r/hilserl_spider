ps -ef|grep run_actor | grep -v grep| awk '{print $2}' | xargs kill -9
ps -ef|grep run_learner | grep -v grep| awk '{print $2}' | xargs kill -9


