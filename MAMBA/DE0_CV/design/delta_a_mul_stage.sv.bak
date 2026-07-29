module delta_a_mul_stage 
    #(
    parameter integer A_SIZE     = 16,
    parameter integer DELTA_SIZE = 16,
    parameter integer OUT_SIZE   = 32,
    parameter integer L_W        = 2,
    parameter integer D_W        = 5,
    parameter integer N_W        = 4
    ) 
    (
    input  logic clk,
    input  logic rst,

    input  logic in_valid,
    input  logic [L_W-1:0] in_l,
    input  logic [D_W-1:0] in_d,
    input  logic [N_W-1:0] in_n,
    input  logic signed [DELTA_SIZE-1:0] delta_value,
    input  logic signed [A_SIZE-1:0]     a_value,

    output logic out_valid,
    output logic [L_W-1:0] out_l,
    output logic [D_W-1:0] out_d,
    output logic [N_W-1:0] out_n,
    output logic signed [OUT_SIZE-1:0] out_data
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            out_l     <= '0;
            out_d     <= '0;
            out_n     <= '0;
            out_data  <= '0;
        end
        else begin
            out_valid <= in_valid;

            if (in_valid) begin
                out_l    <= in_l;
                out_d    <= in_d;
                out_n    <= in_n;
                out_data <= $signed(delta_value) * $signed(a_value);
            end
        end
    end

endmodule
