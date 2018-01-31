FROM sociomantictsunami/dlang:v4
COPY docker/ /docker-tmp
RUN /docker-tmp/build && rm -fr /docker-tmp
