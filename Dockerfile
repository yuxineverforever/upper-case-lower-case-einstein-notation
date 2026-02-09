FROM ubuntu:22.04

WORKDIR /root

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

RUN apt update \
    && apt install -y build-essential git python3 python3-pip python3-dev python3-venv cmake ninja-build


RUN apt update \
    && apt install -y gnupg2 \
    && apt install -y postgresql-common \
    && /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y \
    && apt -y install postgresql-12 postgresql-client-12 libpq-dev postgresql-server-dev-12


RUN git clone https://github.com/tensor-compiler/taco.git \
    && cd taco \
    && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPYTHON=ON \
    && cmake --build build


RUN bash -c "python3 -m venv /root/venv \
    && source /root/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install --upgrade setuptools \
    && pip install --upgrade wheel \
    && pip install numpy torch opt_einsum==3.3.0 torch_geometric scipy psycopg2 pandas"


RUN touch /usr/bin/startup.sh \
    && chmod +x /usr/bin/startup.sh \
    && echo "#!/bin/bash" >> /usr/bin/startup.sh \
    && echo "su - postgres -c 'pg_ctlcluster 12 main start'" >> /usr/bin/startup.sh \
    && echo "su - postgres -c \"psql -c \\\"ALTER USER postgres WITH PASSWORD 'newpassword';\\\"\"" >> /usr/bin/startup.sh \
    && echo "export PYTHONPATH=/root/taco/build/lib:$PYTHONPATH" >> /usr/bin/startup.sh \
    && echo "export PATH=/root/taco/build/bin:$PATH" >> /usr/bin/startup.sh \
    && echo "/bin/bash" >> /usr/bin/startup.sh


RUN touch /root/.bashrc \
    && echo "source /root/venv/bin/activate" >> /root/.bashrc

COPY src/ /root/einsql

ENTRYPOINT ["/usr/bin/startup.sh"]
