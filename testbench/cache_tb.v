`timescale 1ns/1ps

`ifndef CACHE_SIZE_BYTES_CFG
`define CACHE_SIZE_BYTES_CFG 8192
`endif

`ifndef ASSOCIATIVITY_CFG
`define ASSOCIATIVITY_CFG 1
`endif

`ifndef BLOCK_SIZE_BYTES_CFG
`define BLOCK_SIZE_BYTES_CFG 64
`endif

`ifndef REPLACEMENT_POLICY_CFG
`define REPLACEMENT_POLICY_CFG 0
`endif

module cache_tb;
    parameter integer ADDRESS_WIDTH = 32;
    parameter integer CACHE_SIZE_BYTES = `CACHE_SIZE_BYTES_CFG;
    parameter integer ASSOCIATIVITY = `ASSOCIATIVITY_CFG;
    parameter integer BLOCK_SIZE_BYTES = `BLOCK_SIZE_BYTES_CFG;
    parameter integer REPLACEMENT_POLICY = `REPLACEMENT_POLICY_CFG;
    parameter integer LINE_WIDTH = BLOCK_SIZE_BYTES * 8;
    parameter integer WORDS_PER_BLOCK = BLOCK_SIZE_BYTES / 4;
    parameter integer BACKING_LINES = 65536;
    parameter integer NUM_SETS =
        CACHE_SIZE_BYTES / (ASSOCIATIVITY * BLOCK_SIZE_BYTES);
    parameter integer WORD_OFFSET_BITS =
        (WORDS_PER_BLOCK > 1) ? $clog2(WORDS_PER_BLOCK) : 1;

    reg clk;
    reg reset;
    reg req_valid;
    reg req_write;
    reg [ADDRESS_WIDTH-1:0] req_addr;
    reg [31:0] req_wdata;
    wire req_done;
    wire [31:0] resp_rdata;

    wire mem_req_valid;
    wire mem_req_write;
    wire [ADDRESS_WIDTH-1:0] mem_req_addr;
    wire [LINE_WIDTH-1:0] mem_req_wdata;
    wire mem_req_ready;
    reg mem_resp_valid;
    reg [LINE_WIDTH-1:0] mem_resp_rdata;

    wire [63:0] access_count;
    wire [63:0] hit_count;
    wire [63:0] miss_count;

    reg [31:0] backing_address [0:BACKING_LINES-1];
    reg [LINE_WIDTH-1:0] backing_data [0:BACKING_LINES-1];
    reg backing_valid [0:BACKING_LINES-1];
    reg [LINE_WIDTH-1:0] generated_line;
    reg [LINE_WIDTH-1:0] generated_word;
    reg [31:0] expected_read_data;
    reg [31:0] runtime_cache_size;
    reg [31:0] runtime_associativity;
    reg [31:0] runtime_block_size;
    reg [31:0] runtime_policy_code;
    reg [8*1024-1:0] trace_filename;
    reg [8*16-1:0] runtime_policy;
    reg [31:0] trace_address;
    reg [63:0] parsed_trace_address;
    integer trace_operation;
    integer trace_file;
    integer scan_result;
    integer trace_count;
    integer word_number;
    integer init_index;
    integer backing_slot;
    integer policy_present;

    assign mem_req_ready = !mem_resp_valid;

    cache_controller #(
        .ADDRESS_WIDTH(ADDRESS_WIDTH),
        .CACHE_SIZE_BYTES(CACHE_SIZE_BYTES),
        .ASSOCIATIVITY(ASSOCIATIVITY),
        .REPLACEMENT_POLICY(REPLACEMENT_POLICY),
        .BLOCK_SIZE_BYTES(BLOCK_SIZE_BYTES)
    ) dut (
        .clk(clk),
        .reset(reset),
        .req_valid(req_valid),
        .req_write(req_write),
        .req_addr(req_addr),
        .req_wdata(req_wdata),
        .req_done(req_done),
        .resp_rdata(resp_rdata),
        .mem_req_valid(mem_req_valid),
        .mem_req_write(mem_req_write),
        .mem_req_addr(mem_req_addr),
        .mem_req_wdata(mem_req_wdata),
        .mem_req_ready(mem_req_ready),
        .mem_resp_valid(mem_resp_valid),
        .mem_resp_rdata(mem_resp_rdata),
        .access_count(access_count),
        .hit_count(hit_count),
        .miss_count(miss_count)
    );

    always #5 clk = ~clk;

    function integer find_backing_slot;
        input [31:0] address;
        integer first_slot;
        integer probe;
        integer candidate;
        reg found;
        begin
            first_slot = (address / BLOCK_SIZE_BYTES) % BACKING_LINES;
            probe = 0;
            found = 1'b0;
            find_backing_slot = -1;
            while (probe < BACKING_LINES && !found) begin
                candidate = (first_slot + probe) % BACKING_LINES;
                if (!backing_valid[candidate] ||
                    backing_address[candidate] == address) begin
                    find_backing_slot = candidate;
                    found = 1'b1;
                end
                probe = probe + 1;
            end
        end
    endfunction

    always @(posedge clk) begin
        if (reset) begin
            mem_resp_valid <= 1'b0;
            mem_resp_rdata <= {LINE_WIDTH{1'b0}};
        end else begin
            mem_resp_valid <= 1'b0;
            if (mem_req_valid && mem_req_ready) begin
                backing_slot = find_backing_slot(mem_req_addr);
                if (backing_slot < 0) begin
                    $display("ERROR: Testbench backing store is full.");
                    $finish;
                end else if (mem_req_write) begin
                    backing_address[backing_slot] = mem_req_addr;
                    backing_data[backing_slot] = mem_req_wdata;
                    backing_valid[backing_slot] = 1'b1;
                    mem_resp_rdata <= {LINE_WIDTH{1'b0}};
                end else if (backing_valid[backing_slot]) begin
                    mem_resp_rdata <= backing_data[backing_slot];
                end else begin
                    generated_line = {LINE_WIDTH{1'b0}};
                    for (word_number = 0; word_number < WORDS_PER_BLOCK;
                         word_number = word_number + 1) begin
                        generated_word =
                            ((mem_req_addr >> 2) + word_number) ^ 32'h5A5A0000;
                        generated_line = generated_line |
                            (generated_word << (word_number * 32));
                    end
                    backing_address[backing_slot] = mem_req_addr;
                    backing_data[backing_slot] = generated_line;
                    backing_valid[backing_slot] = 1'b1;
                    mem_resp_rdata <= generated_line;
                end
                mem_resp_valid <= 1'b1;
            end
        end
    end

    function [31:0] reference_word;
        input [31:0] address;
        reg [31:0] aligned_address;
        reg [31:0] word_offset;
        reg [LINE_WIDTH-1:0] line_value;
        integer slot;
        begin
            aligned_address = address & ~(BLOCK_SIZE_BYTES - 1);
            word_offset = (address >> 2) & (WORDS_PER_BLOCK - 1);
            slot = find_backing_slot(aligned_address);
            if (slot >= 0 && backing_valid[slot]) begin
                line_value = backing_data[slot];
                reference_word = line_value >> (word_offset * 32);
            end else begin
                reference_word = (address >> 2) ^ 32'h5A5A0000;
            end
        end
    endfunction

    task execute_request;
        input integer operation;
        input [31:0] address;
        input [31:0] write_data;
        begin
            expected_read_data = reference_word(address);
            @(negedge clk);
            req_addr = address;
            req_write = (operation == 1);
            req_wdata = write_data;
            req_valid = 1'b1;
            begin : wait_for_completion
                forever begin
                    @(negedge clk);
                    if (req_done === 1'b1) disable wait_for_completion;
                end
            end
            if (operation == 0 && resp_rdata !== expected_read_data) begin
                $display(
                    "ERROR: Read mismatch at address %08h: expected %08h, received %08h",
                    address, expected_read_data, resp_rdata);
                $finish;
            end
            req_valid = 1'b0;
            req_write = 1'b0;
        end
    endtask

    initial begin
        clk = 1'b0;
        reset = 1'b1;
        req_valid = 1'b0;
        req_write = 1'b0;
        req_addr = {ADDRESS_WIDTH{1'b0}};
        req_wdata = 32'b0;
        mem_resp_valid = 1'b0;
        mem_resp_rdata = {LINE_WIDTH{1'b0}};
        trace_count = 0;
        trace_filename = {8*1024{1'b0}};
        runtime_policy = {8*16{1'b0}};
        runtime_policy_code = 32'b0;
        for (init_index = 0; init_index < BACKING_LINES; init_index = init_index + 1) begin
            backing_valid[init_index] = 1'b0;
            backing_address[init_index] = 32'b0;
            backing_data[init_index] = {LINE_WIDTH{1'b0}};
        end

        if (!$value$plusargs("TRACE_FILE=%s", trace_filename)) begin
            $display("ERROR: Missing required TRACE_FILE plusarg.");
            $finish;
        end
        if (!$value$plusargs("CACHE_SIZE=%d", runtime_cache_size)) begin
            $display("ERROR: Missing required CACHE_SIZE plusarg.");
            $finish;
        end
        if (!$value$plusargs("ASSOCIATIVITY=%d", runtime_associativity)) begin
            $display("ERROR: Missing required ASSOCIATIVITY plusarg.");
            $finish;
        end
        if (!$value$plusargs("BLOCK_SIZE=%d", runtime_block_size)) begin
            $display("ERROR: Missing required BLOCK_SIZE plusarg.");
            $finish;
        end
        if (!$value$plusargs("REPLACEMENT_POLICY_CODE=%d", runtime_policy_code)) begin
            $display("ERROR: Missing required REPLACEMENT_POLICY_CODE plusarg.");
            $finish;
        end
        if (runtime_cache_size != CACHE_SIZE_BYTES ||
            runtime_associativity != ASSOCIATIVITY ||
            runtime_block_size != BLOCK_SIZE_BYTES ||
            runtime_policy_code != REPLACEMENT_POLICY) begin
            $display("ERROR: Runtime geometry differs from elaborated parameters.");
            $finish;
        end
        if ((ASSOCIATIVITY != 1 && ASSOCIATIVITY != 2 && ASSOCIATIVITY != 4) ||
            (REPLACEMENT_POLICY < 0 || REPLACEMENT_POLICY > 2) ||
            CACHE_SIZE_BYTES % (ASSOCIATIVITY * BLOCK_SIZE_BYTES) != 0 ||
            NUM_SETS <= 0 || (NUM_SETS & (NUM_SETS - 1)) != 0) begin
            $display("ERROR: Invalid set-associative cache configuration.");
            $finish;
        end
        policy_present = $value$plusargs("REPLACEMENT_POLICY=%s", runtime_policy);
        if (!policy_present) begin
            $display("ERROR: Missing required REPLACEMENT_POLICY plusarg.");
            $finish;
        end

        $display("CONFIG_TRACE_FILE=%0s", trace_filename);
        $display("CONFIG_CACHE_SIZE=%0d", runtime_cache_size);
        $display("CONFIG_ASSOCIATIVITY=%0d", runtime_associativity);
        $display("CONFIG_BLOCK_SIZE=%0d", runtime_block_size);
        $display("CONFIG_REPLACEMENT_POLICY=%0s", runtime_policy);

        repeat (3) @(negedge clk);
        reset = 1'b0;
        trace_file = $fopen(trace_filename, "r");
        if (trace_file == 0) begin
            $display("ERROR: Could not open trace file: %0s", trace_filename);
            $finish;
        end

        begin : trace_loop
            forever begin
                scan_result = $fscanf(
                    trace_file, "%d %h", trace_operation, parsed_trace_address
                );
                if (scan_result == -1) disable trace_loop;
                if (scan_result != 2) begin
                    $display("ERROR: Malformed trace row %0d.", trace_count + 1);
                    $finish;
                end
                if (parsed_trace_address > 64'h00000000FFFFFFFF) begin
                    $display("ERROR: Trace address exceeds 32-bit address width.");
                    $finish;
                end
                trace_address = parsed_trace_address[31:0];
                if (trace_operation != 0 && trace_operation != 1) begin
                    $display("ERROR: Invalid operation on trace row %0d.", trace_count + 1);
                    $finish;
                end
                if (trace_address[1:0] != 2'b00) begin
                    $display("ERROR: Trace row %0d is not word aligned.", trace_count + 1);
                    $finish;
                end
                execute_request(trace_operation, trace_address, 32'hA5000000 | trace_count);
                trace_count = trace_count + 1;
            end
        end

        $fclose(trace_file);
        repeat (2) @(negedge clk);
        if (access_count != trace_count) begin
            $display("ERROR: Access counter does not match trace length.");
            $finish;
        end
        if (hit_count + miss_count != access_count) begin
            $display("ERROR: Hit/miss counters do not sum to access count.");
            $finish;
        end

        $display("ACCESS_COUNT=%0d", access_count);
        $display("HIT_COUNT=%0d", hit_count);
        $display("MISS_COUNT=%0d", miss_count);
        $finish;
    end
endmodule
