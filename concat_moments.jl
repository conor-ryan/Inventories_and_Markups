#!/usr/bin/env julia

# Concatenate chunk CSV files without external package dependencies.
# Keeps the header from the first file and appends data rows from all chunks.

using Printf

function main()
    input_dir = "SimulatedData"
    output_path = joinpath(input_dir, "moments.csv")

    all_files = sort(readdir(input_dir; join=true))
    chunk_files = filter(f -> occursin(r"moments_\d+\.csv$", basename(f)), all_files)

    isempty(chunk_files) && error("No chunk files found in SimulatedData/")

    rows_written = 0
    header_line = nothing

    open(output_path, "w") do fout
        for f in chunk_files
            first_line_in_file = true
            open(f, "r") do fin
                for line in eachline(fin)
                    if first_line_in_file
                        if header_line === nothing
                            header_line = line
                            println(fout, line)
                        elseif line != header_line
                            error("Header mismatch in $(f)")
                        end
                        first_line_in_file = false
                        continue
                    end
                    println(fout, line)
                    rows_written += 1
                end
            end
        end
    end

    @printf("Concatenated %d files -> %s (%d rows)\n", length(chunk_files), output_path, rows_written)
end

main()
