.PHONY: install upgrade nltk install-kernel-syftr remove-kernel-syftr mypy pre-commit-run unit-tests functional-tests e2e-tests tests aws-login submit- ray-submit- ray-stop-job ray-experiment- ray-cancel-jobs check tunnel

export AWS_PROFILE=rd
OUTFILE := syftr.txt

install:
	@uv pip install -e ".[dev]"
	@pre-commit install || true
	@uv lock

upgrade:
	@uv pip install --upgrade -e ".[dev]"
	@pre-commit install || true
	@uv lock

nltk:
	python -m syftr.startup

install-kernel-syftr:
	@python -m ipykernel install --user --name=syftr

remove-kernel-syftr:
	@jupyter kernelspec remove -y syftr || true

mypy:
	@uv run --frozen mypy . --incremental

pre-commit-run:
	@pre-commit run --all-files

unit-tests:
	@uv run pytest -svv tests/unit/
functional-tests:
	@uv run pytest -svv tests/functional/

e2e-tests:
	@uv run pytest -svv tests/e2e/

tests: unit-tests functional-tests e2e-tests

# Use like `make submit-hotpot-toy`, which submits the study defined in studies/hotpot-toy.yaml
submit-%: studies/%.yaml
	@python -m syftr.ray.submit --study-config $< 2>&1 | tee $(OUTFILE)

aws-login:
	@test -n "$(shell find ~/.aws/sso/cache/ -type f -mmin -480)" || yawsso login --profile rd
	@python -m syftr.amazon

# Use like `make ray-submit-hotpot`, which submits study defined in studies/hotpot.yaml
ray-submit-%: studies/%.yaml
	@PYTHONUNBUFFERED=1 python -m syftr.ray.submit --remote --study-config $<

ray-stop-job:
	@python -m syftr.ray.stop --remote --id="$(id)"

ray-experiment-%:
	@PYTHONUNBUFFERED=1 python -m syftr.scripts.experiments.$* --remote

ray-cancel-jobs:
	@PYTHONUNBUFFERED=1 python -m syftr.scripts.cancel_jobs --substring="$(substring)"

tunnel:
	ssh -NT \
		-o ServerAliveInterval=10 \
		-o ServerAliveCountMax=3 \
		-o TCPKeepAlive=yes \
		-o ExitOnForwardFailure=true \
		-o ControlMaster=auto \
		-o ControlPath=/tmp/ssh-tunnel-%r@%h:%p \
		-o ControlPersist=yes \
		-L 127.0.0.1:10001:localhost:10001 \
		-L 127.0.0.1:3000:localhost:3000 \
		-L 127.0.0.1:5432:localhost:5432 \
		-L 127.0.0.1:16686:localhost:16686 \
		-L 127.0.0.1:8265:localhost:8265 \
		jump

tunnel-embeds:
	ssh -J jump \
	    -o ServerAliveInterval=10 \
	    -o ServerAliveCountMax=3 \
	    -o TCPKeepAlive=yes \
	    -o ExitOnForwardFailure=true \
            -o ControlMaster=auto \
            -o ControlPath=/tmp/embeds-tunnel-%r@%h:%p \
            -o ControlPersist=yes \
	    -L 127.0.0.1:9001:localhost:8001 \
	    -L 127.0.0.1:9002:localhost:8002 \
	    -L 127.0.0.1:9003:localhost:8003 \
	    -L 127.0.0.1:9004:localhost:8004 \
	    -L 127.0.0.1:9005:localhost:8005 \
	    -L 127.0.0.1:9006:localhost:8006 \
	    -L 127.0.0.1:9007:localhost:8007 \
	    -L 127.0.0.1:9008:localhost:8008 \
	    -L 127.0.0.1:9009:localhost:8009 \
	    -L 127.0.0.1:9010:localhost:8010 \
	    -L 127.0.0.1:9011:localhost:8011 \
	    -L 127.0.0.1:9012:localhost:8012 \
	    -L 127.0.0.1:9013:localhost:8013 \
	    -L 127.0.0.1:9014:localhost:8014 \
	    -L 127.0.0.1:9015:localhost:8015 \
	    -L 127.0.0.1:9016:localhost:8016 \
	    -L 127.0.0.1:9017:localhost:8017 \
	    -L 127.0.0.1:9018:localhost:8018 \
	    gpu-293

reverse-tunnel:
	ssh -J jump \
	    -o ServerAliveInterval=10 \
	    -o ServerAliveCountMax=10 \
	    -o TCPKeepAlive=yes \
	    -o ExitOnForwardFailure=true \
            -o ControlMaster=auto \
            -o ControlPath=/tmp/reverse-tunnel-%r@%h:%p \
            -o ControlPersist=yes \
	    -R 8443:staging.datarobot.com:443 \
	    gpu-293

check:
	@python -m syftr.scripts.system_check
