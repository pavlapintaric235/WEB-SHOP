FROM public.ecr.aws/lambda/python:3.13

COPY experiments/opencv5_guided_scan/aws_pilot/requirements-lambda.txt /tmp/requirements-lambda.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: \
    -r /tmp/requirements-lambda.txt --target "${LAMBDA_TASK_ROOT}"

COPY experiments/opencv5_guided_scan/capture_gate.py "${LAMBDA_TASK_ROOT}/capture_gate.py"
COPY experiments/opencv5_guided_scan/aws_pilot/handler.py "${LAMBDA_TASK_ROOT}/handler.py"

CMD ["handler.lambda_handler"]
