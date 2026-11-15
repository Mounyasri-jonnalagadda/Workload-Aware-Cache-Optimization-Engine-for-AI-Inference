`timescale 1ns/1ps

module cache_memory #(
    parameter integer ADDRESS_WIDTH = 32,
    parameter integer CACHE_SIZE_BYTES = 8192,
    parameter integer ASSOCIATIVITY = 1,
    parameter integer BLOCK_SIZE_BYTES = 64,
    parameter integer NUM_SETS = CACHE_SIZE_BYTES / (ASSOCIATIVITY * BLOCK_SIZE_BYTES),
    parameter integer INDEX_BITS = (NUM_SETS > 1) ? $clog2(NUM_SETS) : 0,
    parameter integer INDEX_WIDTH = (INDEX_BITS > 0) ? INDEX_BITS : 1,
    parameter integer OFFSET_BITS = $clog2(BLOCK_SIZE_BYTES),
    parameter integer TAG_WIDTH = ADDRESS_WIDTH - INDEX_BITS - OFFSET_BITS,
    parameter integer LINE_WIDTH = BLOCK_SIZE_BYTES * 8,
    parameter integer TOTAL_LINES = NUM_SETS * ASSOCIATIVITY
) (
    input  wire                                  clk,
    input  wire                                  reset,
    input  wire [INDEX_WIDTH-1:0]                read_set,
    output reg  [ASSOCIATIVITY-1:0]             read_valid,
    output reg  [ASSOCIATIVITY*TAG_WIDTH-1:0]   read_tags,
    output reg  [ASSOCIATIVITY*LINE_WIDTH-1:0]  read_lines,
    input  wire                                  write_enable,
    input  wire [INDEX_WIDTH-1:0]                write_set,
    input  wire [((ASSOCIATIVITY > 1) ? $clog2(ASSOCIATIVITY) : 1)-1:0] write_way,
    input  wire [TAG_WIDTH-1:0]                  write_tag,
    input  wire [LINE_WIDTH-1:0]                 write_line
);
    reg valid_bits [0:TOTAL_LINES-1];
    reg [TAG_WIDTH-1:0] tag_array [0:TOTAL_LINES-1];
    reg [LINE_WIDTH-1:0] data_array [0:TOTAL_LINES-1];
    integer line_number;
    integer way_number;
    integer read_line_number;
    integer write_line_number;

    always @* begin
        read_valid = {ASSOCIATIVITY{1'b0}};
        read_tags = {(ASSOCIATIVITY*TAG_WIDTH){1'b0}};
        read_lines = {(ASSOCIATIVITY*LINE_WIDTH){1'b0}};
        for (way_number = 0; way_number < ASSOCIATIVITY; way_number = way_number + 1) begin
            read_line_number = read_set * ASSOCIATIVITY + way_number;
            read_valid[way_number] = valid_bits[read_line_number];
            read_tags[way_number*TAG_WIDTH +: TAG_WIDTH] = tag_array[read_line_number];
            read_lines[way_number*LINE_WIDTH +: LINE_WIDTH] = data_array[read_line_number];
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            for (line_number = 0; line_number < TOTAL_LINES; line_number = line_number + 1) begin
                valid_bits[line_number] <= 1'b0;
            end
        end else if (write_enable) begin
            write_line_number = write_set * ASSOCIATIVITY + write_way;
            valid_bits[write_line_number] <= 1'b1;
            tag_array[write_line_number] <= write_tag;
            data_array[write_line_number] <= write_line;
        end
    end
endmodule
