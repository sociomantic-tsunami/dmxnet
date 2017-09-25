FROM sociomantictsunami/dlang:v3
COPY docker/ /docker-tmp
RUN /docker-tmp/build && rm -fr /docker-tmp
