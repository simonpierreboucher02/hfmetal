function(hfm_compile_metal_shaders target shader_dir output_dir)
    # Check if Metal shader compiler is available
    execute_process(
        COMMAND xcrun -sdk macosx --find metal
        OUTPUT_VARIABLE METAL_COMPILER
        ERROR_QUIET
        RESULT_VARIABLE METAL_FOUND
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT METAL_FOUND EQUAL 0)
        message(WARNING "Metal shader compiler not found (need full Xcode, not just CLT). "
                        "Shaders will be compiled at runtime from source.")

        # Copy .metal source files to build dir for runtime compilation
        file(MAKE_DIRECTORY ${output_dir})
        file(GLOB METAL_SOURCES "${shader_dir}/*.metal")
        foreach(shader ${METAL_SOURCES})
            get_filename_component(shader_name ${shader} NAME)
            configure_file(${shader} "${output_dir}/${shader_name}" COPYONLY)
        endforeach()

        target_compile_definitions(${target} PRIVATE
            HFM_METAL_SHADER_DIR="${output_dir}"
            HFM_METAL_RUNTIME_COMPILE=1
        )
        return()
    endif()

    file(MAKE_DIRECTORY ${output_dir})
    file(GLOB METAL_SOURCES "${shader_dir}/*.metal")

    set(AIR_FILES)
    foreach(shader ${METAL_SOURCES})
        get_filename_component(shader_name ${shader} NAME_WE)
        set(air_file "${output_dir}/${shader_name}.air")
        add_custom_command(
            OUTPUT ${air_file}
            COMMAND xcrun -sdk macosx metal -c ${shader} -o ${air_file}
                -std=metal3.0 -O2
            DEPENDS ${shader}
            COMMENT "Compiling Metal shader: ${shader_name}.metal"
        )
        list(APPEND AIR_FILES ${air_file})
    endforeach()

    set(METALLIB_FILE "${output_dir}/hfmetal.metallib")
    add_custom_command(
        OUTPUT ${METALLIB_FILE}
        COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${METALLIB_FILE}
        DEPENDS ${AIR_FILES}
        COMMENT "Linking Metal library: hfmetal.metallib"
    )

    add_custom_target(${target}_metal_shaders ALL DEPENDS ${METALLIB_FILE})
    add_dependencies(${target} ${target}_metal_shaders)

    target_compile_definitions(${target} PRIVATE
        HFM_METALLIB_PATH="${METALLIB_FILE}"
    )
endfunction()
