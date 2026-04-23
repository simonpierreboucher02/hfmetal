include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(TARGETS hfmetal
    EXPORT hfmetalTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/hfm
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT hfmetalTargets
    FILE hfmetalTargets.cmake
    NAMESPACE hfm::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hfmetal
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hfmetalConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
