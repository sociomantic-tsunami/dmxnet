FROM sociomantictsunami/dlang:v5
COPY docker/ /docker-tmp
RUN /docker-tmp/build && rm -fr /docker-tmp
