`timescale 1ns/1ps

module cache_controller #(
    parameter integer ADDRESS_WIDTH = 32,
    parameter integer CACHE_SIZE_BYTES = 8192,
    parameter integer ASSOCIATIVITY = 1,
    parameter integer BLOCK_SIZE_BYTES = 64,
    parameter integer REPLACEMENT_POLICY = 0,
    parameter integer SAFE_ASSOCIATIVITY =
        (ASSOCIATIVITY > 0) ? ASSOCIATIVITY : 1,
    parameter integer SAFE_BLOCK_SIZE_BYTES =
        (BLOCK_SIZE_BYTES > 0) ? BLOCK_SIZE_BYTES : 1,
    parameter integer NUM_SETS =
        CACHE_SIZE_BYTES / (SAFE_ASSOCIATIVITY * SAFE_BLOCK_SIZE_BYTES),
    parameter integer TOTAL_LINES = NUM_SETS * ASSOCIATIVITY,
    parameter integer WORDS_PER_BLOCK = BLOCK_SIZE_BYTES / 4,
    parameter integer LINE_WIDTH = BLOCK_SIZE_BYTES * 8,
    parameter integer WORD_OFFSET_BITS =
        (WORDS_PER_BLOCK > 1) ? $clog2(WORDS_PER_BLOCK) : 1,
    parameter integer INDEX_BITS = (NUM_SETS > 1) ? $clog2(NUM_SETS) : 0,
    parameter integer INDEX_WIDTH = (INDEX_BITS > 0) ? INDEX_BITS : 1,
    parameter integer OFFSET_BITS = $clog2(SAFE_BLOCK_SIZE_BYTES),
    parameter integer TAG_WIDTH = ADDRESS_WIDTH - INDEX_BITS - OFFSET_BITS,
    parameter integer WAY_WIDTH =
        (ASSOCIATIVITY > 1) ? $clog2(ASSOCIATIVITY) : 1,
    parameter integer LRU_AGE_WIDTH =
        (ASSOCIATIVITY > 1) ? $clog2(ASSOCIATIVITY) : 1
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire                     req_valid,
    input  wire                     req_write,
    input  wire [ADDRESS_WIDTH-1:0] req_addr,
    input  wire [31:0]              req_wdata,
    output reg                      req_done,
    output reg  [31:0]              resp_rdata,

    output reg                      mem_req_valid,
    output reg                      mem_req_write,
    output reg  [ADDRESS_WIDTH-1:0] mem_req_addr,
    output reg  [LINE_WIDTH-1:0]    mem_req_wdata,
    input  wire                     mem_req_ready,
    input  wire                     mem_resp_valid,
    input  wire [LINE_WIDTH-1:0]    mem_resp_rdata,

    output wire [63:0]              access_count,
    output wire [63:0]              hit_count,
    output wire [63:0]              miss_count
);
    localparam [2:0] STATE_IDLE = 3'd0;
    localparam [2:0] STATE_MEM_READ_REQUEST = 3'd1;
    localparam [2:0] STATE_MEM_READ_RESPONSE = 3'd2;
    localparam [2:0] STATE_MEM_WRITE_REQUEST = 3'd3;
    localparam [2:0] STATE_MEM_WRITE_RESPONSE = 3'd4;

    reg [2:0] state;
    reg [ADDRESS_WIDTH-1:0] pending_addr;
    reg [31:0] pending_wdata;
    reg pending_write;
    reg [WORD_OFFSET_BITS-1:0] pending_word_offset;
    reg [TAG_WIDTH-1:0] pending_tag;
    reg [INDEX_WIDTH-1:0] pending_set;
    reg [WAY_WIDTH-1:0] pending_way;
    reg [LINE_WIDTH-1:0] pending_line;

    wire [TAG_WIDTH-1:0] lookup_tag;
    wire [INDEX_WIDTH-1:0] lookup_set;
    wire [OFFSET_BITS-1:0] lookup_offset;
    wire [ADDRESS_WIDTH-1:0] lookup_addr;
    wire [ASSOCIATIVITY-1:0] cache_valid_vec;
    wire [ASSOCIATIVITY*TAG_WIDTH-1:0] cache_tags;
    wire [ASSOCIATIVITY*LINE_WIDTH-1:0] cache_lines;
    reg cache_hit;
    reg [WAY_WIDTH-1:0] hit_way;
    reg [LINE_WIDTH-1:0] cache_line;
    reg invalid_way_found;
    reg [WAY_WIDTH-1:0] first_invalid_way;
    reg [WAY_WIDTH-1:0] victim_way;
    reg [LRU_AGE_WIDTH-1:0] oldest_age;
    reg [WORD_OFFSET_BITS-1:0] lookup_word_offset;
    wire access_event;
    wire hit_event;
    wire miss_event;

    reg cache_write_enable;
    reg [INDEX_WIDTH-1:0] cache_write_set;
    reg [WAY_WIDTH-1:0] cache_write_way;
    reg [TAG_WIDTH-1:0] cache_write_tag;
    reg [LINE_WIDTH-1:0] cache_write_line;
    reg [LINE_WIDTH-1:0] modified_line;

    reg [WAY_WIDTH-1:0] fifo_pointer [0:NUM_SETS-1];
    reg [LRU_AGE_WIDTH-1:0] lru_age [0:TOTAL_LINES-1];
    reg [15:0] random_state;
    integer hit_scan;
    integer victim_scan;
    integer set_number;
    integer way_number;

    assign lookup_addr = (state == STATE_IDLE) ? req_addr : pending_addr;
    assign access_event = (state == STATE_IDLE) && req_valid;
    assign hit_event = access_event && cache_hit;
    assign miss_event = access_event && !cache_hit;

    address_decoder #(
        .ADDRESS_WIDTH(ADDRESS_WIDTH),
        .CACHE_SIZE_BYTES(CACHE_SIZE_BYTES),
        .ASSOCIATIVITY(ASSOCIATIVITY),
        .BLOCK_SIZE_BYTES(BLOCK_SIZE_BYTES),
        .NUM_SETS(NUM_SETS),
        .INDEX_BITS(INDEX_BITS),
        .INDEX_WIDTH(INDEX_WIDTH),
        .OFFSET_BITS(OFFSET_BITS),
        .TAG_WIDTH(TAG_WIDTH)
    ) decoder (
        .address(lookup_addr),
        .tag(lookup_tag),
        .index(lookup_set),
        .offset(lookup_offset)
    );

    cache_memory #(
        .ADDRESS_WIDTH(ADDRESS_WIDTH),
        .CACHE_SIZE_BYTES(CACHE_SIZE_BYTES),
        .ASSOCIATIVITY(ASSOCIATIVITY),
        .BLOCK_SIZE_BYTES(BLOCK_SIZE_BYTES),
        .NUM_SETS(NUM_SETS),
        .INDEX_BITS(INDEX_BITS),
        .INDEX_WIDTH(INDEX_WIDTH),
        .OFFSET_BITS(OFFSET_BITS),
        .TAG_WIDTH(TAG_WIDTH),
        .LINE_WIDTH(LINE_WIDTH),
        .TOTAL_LINES(TOTAL_LINES)
    ) data_store (
        .clk(clk),
        .reset(reset),
        .read_set(lookup_set),
        .read_valid(cache_valid_vec),
        .read_tags(cache_tags),
        .read_lines(cache_lines),
        .write_enable(cache_write_enable),
        .write_set(cache_write_set),
        .write_way(cache_write_way),
        .write_tag(cache_write_tag),
        .write_line(cache_write_line)
    );

    performance_counter counters (
        .clk(clk),
        .reset(reset),
        .access_event(access_event),
        .hit_event(hit_event),
        .miss_event(miss_event),
        .access_count(access_count),
        .hit_count(hit_count),
        .miss_count(miss_count)
    );

    always @* begin
        cache_hit = 1'b0;
        hit_way = {WAY_WIDTH{1'b0}};
        cache_line = {LINE_WIDTH{1'b0}};
        for (hit_scan = 0; hit_scan < ASSOCIATIVITY; hit_scan = hit_scan + 1) begin
            if (cache_valid_vec[hit_scan] &&
                cache_tags[hit_scan*TAG_WIDTH +: TAG_WIDTH] == lookup_tag &&
                !cache_hit) begin
                cache_hit = 1'b1;
                hit_way = hit_scan;
                cache_line = cache_lines[hit_scan*LINE_WIDTH +: LINE_WIDTH];
            end
        end
    end

    always @* begin
        invalid_way_found = 1'b0;
        first_invalid_way = {WAY_WIDTH{1'b0}};
        victim_way = {WAY_WIDTH{1'b0}};
        oldest_age = {LRU_AGE_WIDTH{1'b0}};
        for (victim_scan = 0; victim_scan < ASSOCIATIVITY; victim_scan = victim_scan + 1) begin
            if (!cache_valid_vec[victim_scan] && !invalid_way_found) begin
                invalid_way_found = 1'b1;
                first_invalid_way = victim_scan;
            end
        end
        if (invalid_way_found) begin
            victim_way = first_invalid_way;
        end else begin
            case (REPLACEMENT_POLICY)
                0: begin
                    victim_way = {WAY_WIDTH{1'b0}};
                    oldest_age = lru_age[lookup_set*ASSOCIATIVITY];
                    for (victim_scan = 1; victim_scan < ASSOCIATIVITY; victim_scan = victim_scan + 1) begin
                        if (lru_age[lookup_set*ASSOCIATIVITY + victim_scan] > oldest_age) begin
                            oldest_age = lru_age[lookup_set*ASSOCIATIVITY + victim_scan];
                            victim_way = victim_scan;
                        end
                    end
                end
                1: victim_way = fifo_pointer[lookup_set];
                2: begin
                    if (ASSOCIATIVITY == 2)
                        victim_way = random_state[0];
                    else if (ASSOCIATIVITY == 4)
                        victim_way = random_state[1:0];
                    else
                        victim_way = {WAY_WIDTH{1'b0}};
                end
                default: victim_way = {WAY_WIDTH{1'b0}};
            endcase
        end
    end

    task update_lru;
        input [INDEX_WIDTH-1:0] set_index;
        input [WAY_WIDTH-1:0] accessed_way;
        integer age_way;
        reg [LRU_AGE_WIDTH-1:0] accessed_age;
        begin
            accessed_age = lru_age[set_index*ASSOCIATIVITY + accessed_way];
            for (age_way = 0; age_way < ASSOCIATIVITY; age_way = age_way + 1) begin
                if (age_way == accessed_way) begin
                    lru_age[set_index*ASSOCIATIVITY + age_way] <= {LRU_AGE_WIDTH{1'b0}};
                end else if (lru_age[set_index*ASSOCIATIVITY + age_way] < accessed_age) begin
                    lru_age[set_index*ASSOCIATIVITY + age_way] <=
                        lru_age[set_index*ASSOCIATIVITY + age_way] + 1'b1;
                end
            end
        end
    endtask

    function [LINE_WIDTH-1:0] replace_word;
        input [LINE_WIDTH-1:0] line_value;
        input [WORD_OFFSET_BITS-1:0] word_offset;
        input [31:0] word_value;
        begin
            replace_word = line_value;
            replace_word[word_offset * 32 +: 32] = word_value;
        end
    endfunction

    function [31:0] select_word;
        input [LINE_WIDTH-1:0] line_value;
        input [WORD_OFFSET_BITS-1:0] word_offset;
        begin
            select_word = line_value[word_offset * 32 +: 32];
        end
    endfunction

    always @* begin
        lookup_word_offset = lookup_offset >> 2;
        cache_write_enable = 1'b0;
        cache_write_set = lookup_set;
        cache_write_way = hit_way;
        cache_write_tag = lookup_tag;
        cache_write_line = cache_line;
        modified_line = cache_line;

        mem_req_valid = 1'b0;
        mem_req_write = 1'b0;
        mem_req_addr = pending_addr & ~(BLOCK_SIZE_BYTES - 1);
        mem_req_wdata = pending_line;

        if (state == STATE_IDLE && req_valid && cache_hit && req_write) begin
            modified_line = replace_word(cache_line, lookup_word_offset, req_wdata);
            cache_write_enable = 1'b1;
            cache_write_line = modified_line;
        end else if (state == STATE_MEM_READ_RESPONSE && mem_resp_valid) begin
            if (pending_write) begin
                modified_line = replace_word(
                    mem_resp_rdata, pending_word_offset, pending_wdata
                );
                cache_write_line = modified_line;
            end else begin
                cache_write_line = mem_resp_rdata;
            end
            cache_write_enable = 1'b1;
            cache_write_set = pending_set;
            cache_write_way = pending_way;
            cache_write_tag = pending_tag;
        end

        if (state == STATE_MEM_READ_REQUEST) begin
            mem_req_valid = 1'b1;
        end else if (state == STATE_MEM_WRITE_REQUEST) begin
            mem_req_valid = 1'b1;
            mem_req_write = 1'b1;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            state <= STATE_IDLE;
            pending_addr <= {ADDRESS_WIDTH{1'b0}};
            pending_wdata <= 32'b0;
            pending_write <= 1'b0;
            pending_word_offset <= {WORD_OFFSET_BITS{1'b0}};
            pending_tag <= {TAG_WIDTH{1'b0}};
            pending_set <= {INDEX_WIDTH{1'b0}};
            pending_way <= {WAY_WIDTH{1'b0}};
            pending_line <= {LINE_WIDTH{1'b0}};
            req_done <= 1'b0;
            resp_rdata <= 32'b0;
            random_state <= 16'h0001;
            for (set_number = 0; set_number < NUM_SETS; set_number = set_number + 1) begin
                fifo_pointer[set_number] <= {WAY_WIDTH{1'b0}};
                for (way_number = 0; way_number < ASSOCIATIVITY; way_number = way_number + 1) begin
                    lru_age[set_number*ASSOCIATIVITY + way_number] <= way_number;
                end
            end
        end else begin
            req_done <= 1'b0;
            case (state)
                STATE_IDLE: begin
                    if (req_valid) begin
                        pending_addr <= req_addr;
                        pending_wdata <= req_wdata;
                        pending_write <= req_write;
                        pending_word_offset <= lookup_word_offset;
                        pending_tag <= lookup_tag;
                        pending_set <= lookup_set;
                        pending_way <= cache_hit ? hit_way : victim_way;
                        if (cache_hit) begin
                            update_lru(lookup_set, hit_way);
                            if (req_write) begin
                                pending_line <= replace_word(
                                    cache_line, lookup_word_offset, req_wdata
                                );
                                state <= STATE_MEM_WRITE_REQUEST;
                            end else begin
                                resp_rdata <= select_word(cache_line, lookup_word_offset);
                                req_done <= 1'b1;
                            end
                        end else begin
                            state <= STATE_MEM_READ_REQUEST;
                        end
                    end
                end

                STATE_MEM_READ_REQUEST: begin
                    if (mem_req_ready) state <= STATE_MEM_READ_RESPONSE;
                end

                STATE_MEM_READ_RESPONSE: begin
                    if (mem_resp_valid) begin
                        update_lru(pending_set, pending_way);
                        if (REPLACEMENT_POLICY == 1) begin
                            if (pending_way == ASSOCIATIVITY - 1)
                                fifo_pointer[pending_set] <= {WAY_WIDTH{1'b0}};
                            else
                                fifo_pointer[pending_set] <= pending_way + 1'b1;
                        end
                        random_state <= {
                            random_state[14:0],
                            random_state[15] ^ random_state[13] ^
                            random_state[12] ^ random_state[10]
                        };
                        if (pending_write) begin
                            pending_line <= replace_word(
                                mem_resp_rdata, pending_word_offset, pending_wdata
                            );
                            state <= STATE_MEM_WRITE_REQUEST;
                        end else begin
                            resp_rdata <= select_word(mem_resp_rdata, pending_word_offset);
                            req_done <= 1'b1;
                            state <= STATE_IDLE;
                        end
                    end
                end

                STATE_MEM_WRITE_REQUEST: begin
                    if (mem_req_ready) state <= STATE_MEM_WRITE_RESPONSE;
                end

                STATE_MEM_WRITE_RESPONSE: begin
                    if (mem_resp_valid) begin
                        req_done <= 1'b1;
                        resp_rdata <= 32'b0;
                        state <= STATE_IDLE;
                    end
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (ADDRESS_WIDTH != 32 || CACHE_SIZE_BYTES <= 0 ||
            (ASSOCIATIVITY != 1 && ASSOCIATIVITY != 2 && ASSOCIATIVITY != 4) ||
            (REPLACEMENT_POLICY < 0 || REPLACEMENT_POLICY > 2) ||
            BLOCK_SIZE_BYTES < 4 || (BLOCK_SIZE_BYTES % 4) != 0 ||
            (BLOCK_SIZE_BYTES & (BLOCK_SIZE_BYTES - 1)) != 0 ||
            CACHE_SIZE_BYTES < BLOCK_SIZE_BYTES * ASSOCIATIVITY ||
            CACHE_SIZE_BYTES % (BLOCK_SIZE_BYTES * ASSOCIATIVITY) != 0 ||
            (NUM_SETS & (NUM_SETS - 1)) != 0 || TAG_WIDTH <= 0) begin
            $fatal(1, "Invalid set-associative cache geometry or replacement policy.");
        end
    end
`endif
endmodule
