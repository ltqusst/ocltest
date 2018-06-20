# https://stackoverflow.com/questions/11813271/embed-resources-eg-shader-code-images-into-executable-library-with-cmake
# Creates C resources file from files in given directory
function(create_resources bins output)
    # Create empty output file
    file(WRITE ${output} "/***********************************************\n")
    file(APPEND ${output} "  DO NOT MODIFY, automatically generated from files:\n")
    foreach(bin ${bins})
        file(APPEND ${output} "   ${bin}\n")
    endforeach()
    file(APPEND ${output} "************************************************/\n")
    # Iterate through input files
    foreach(bin ${bins})
        # Get short filename
        string(REGEX MATCH "([^/]+)$" filename ${bin})
        # Replace filename spaces & extension separator for C compatibility
        string(REGEX REPLACE "\\.| |-" "_" filename ${filename})
        # Read hex data from file
        file(READ ${bin} filedata HEX)
        # Convert hex data for C compatibility
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," filedata ${filedata})
        # Append data to output file
        file(APPEND ${output} "const unsigned char ${filename}[] = {${filedata}0x00};\nconst unsigned ${filename}_size = sizeof(${filename});\n")
    endforeach()
endfunction()

# Collect input files
# file(GLOB bins ${dir}/*)

message("create_resources invoked with src=${src}, dst=${dst}")

create_resources(${src}  ${dst})

