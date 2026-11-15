`timescale 1ns/1ps

module performance_counter #(
    parameter integer COUNTER_WIDTH = 64
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire                     access_event,
    input  wire                     hit_event,
    input  wire                     miss_event,
    output reg  [COUNTER_WIDTH-1:0] access_count,
    output reg  [COUNTER_WIDTH-1:0] hit_count,
    output reg  [COUNTER_WIDTH-1:0] miss_count
);
    always @(posedge clk) begin
        if (reset) begin
            access_count <= {COUNTER_WIDTH{1'b0}};
            hit_count <= {COUNTER_WIDTH{1'b0}};
            miss_count <= {COUNTER_WIDTH{1'b0}};
        end else begin
            if (access_event) access_count <= access_count + 1'b1;
            if (hit_event) hit_count <= hit_count + 1'b1;
            if (miss_event) miss_count <= miss_count + 1'b1;
        end
    end
endmodule
