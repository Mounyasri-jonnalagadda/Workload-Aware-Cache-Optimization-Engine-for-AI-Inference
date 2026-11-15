`timescale 1ns/1ps

module address_decoder #(
    parameter integer ADDRESS_WIDTH = 32,
    parameter integer CACHE_SIZE_BYTES = 8192,
    parameter integer ASSOCIATIVITY = 1,
    parameter integer BLOCK_SIZE_BYTES = 64,
    parameter integer SAFE_ASSOCIATIVITY =
        (ASSOCIATIVITY > 0) ? ASSOCIATIVITY : 1,
    parameter integer SAFE_BLOCK_SIZE_BYTES =
        (BLOCK_SIZE_BYTES > 0) ? BLOCK_SIZE_BYTES : 1,
    parameter integer NUM_SETS =
        CACHE_SIZE_BYTES / (SAFE_ASSOCIATIVITY * SAFE_BLOCK_SIZE_BYTES),
    parameter integer OFFSET_BITS = $clog2(SAFE_BLOCK_SIZE_BYTES),
    parameter integer INDEX_BITS = (NUM_SETS > 1) ? $clog2(NUM_SETS) : 0,
    parameter integer TAG_BITS = ADDRESS_WIDTH - OFFSET_BITS - INDEX_BITS,
    parameter integer OFFSET_WIDTH = (OFFSET_BITS > 0) ? OFFSET_BITS : 1,
    parameter integer INDEX_WIDTH = (INDEX_BITS > 0) ? INDEX_BITS : 1,
    parameter integer TAG_WIDTH = (TAG_BITS > 0) ? TAG_BITS : 1
) (
    input  wire [ADDRESS_WIDTH-1:0] address,
    output wire [TAG_WIDTH-1:0] tag,
    output wire [INDEX_WIDTH-1:0] index,
    output wire [OFFSET_WIDTH-1:0] offset
);
    assign tag = address >> (OFFSET_BITS + INDEX_BITS);
    assign index = (address >> OFFSET_BITS) & (NUM_SETS - 1);
    assign offset = address & (SAFE_BLOCK_SIZE_BYTES - 1);

`ifndef SYNTHESIS
    initial begin
        if (ADDRESS_WIDTH != 32 || CACHE_SIZE_BYTES <= 0 ||
            ASSOCIATIVITY <= 0 || BLOCK_SIZE_BYTES < 4 ||
            CACHE_SIZE_BYTES % (SAFE_ASSOCIATIVITY * SAFE_BLOCK_SIZE_BYTES) != 0 ||
            (SAFE_BLOCK_SIZE_BYTES & (SAFE_BLOCK_SIZE_BYTES - 1)) != 0 ||
            NUM_SETS <= 0 || (NUM_SETS & (NUM_SETS - 1)) != 0 ||
            TAG_BITS <= 0) begin
            $fatal(1, "Invalid set-associative cache geometry.");
        end
    end
`endif
endmodule
