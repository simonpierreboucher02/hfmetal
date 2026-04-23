function(hfm_enable_sanitizers target)
    target_compile_options(${target} PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(${target} PRIVATE
        -fsanitize=address,undefined
    )
endfunction()
