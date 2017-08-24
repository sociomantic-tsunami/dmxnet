FROM sociomantictsunami/dlang:v2
COPY docker/ /docker-tmp
RUN /docker-tmp/build && rm -fr /docker-tmp
