FROM gh-cli

# python3 for serving the lecture pages
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /course
COPY . /course

EXPOSE 8099
# gh-cli set ENTRYPOINT to gh; reset it so the default command runs the server
ENTRYPOINT []
CMD ["python3", "-m", "http.server", "8099"]
