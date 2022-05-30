FROM migrate/migrate

ADD https://github.com/ufoscout/docker-compose-wait/releases/download/2.9.0/wait /wait
RUN chmod +x /wait

COPY entrypoint.sh /usr/local/bin
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT [ "entrypoint.sh" ]