# Included by `cmake --install` (component `docs`) with MANTA_DOCS_HTML and
# MANTA_DOCS_SOURCE_COPY set: copy the built HTML into the checkout. See the
# install rules beside the `docs` target in MantaTools.cmake for why this exists.

if(NOT EXISTS "${MANTA_DOCS_HTML}/index.html")
  message(STATUS "Docs not built, so not copied to ${MANTA_DOCS_SOURCE_COPY}: "
                 "build the `docs` target first")
  return()
endif()
if(NOT "$ENV{DESTDIR}" STREQUAL "")
  message(STATUS "DESTDIR is set, so not copying the docs to ${MANTA_DOCS_SOURCE_COPY}")
  return()
endif()

# Replace rather than overlay, so a page that has since been deleted does not
# survive -- but only a directory Sphinx wrote, which .buildinfo marks. A cache
# value naming anything else is copied into and never emptied.
if(EXISTS "${MANTA_DOCS_SOURCE_COPY}/.buildinfo")
  file(REMOVE_RECURSE "${MANTA_DOCS_SOURCE_COPY}")
endif()
message(STATUS "Installing: ${MANTA_DOCS_SOURCE_COPY}")
file(COPY "${MANTA_DOCS_HTML}/" DESTINATION "${MANTA_DOCS_SOURCE_COPY}")
