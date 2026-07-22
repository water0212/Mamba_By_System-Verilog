module stream_delay 
	#(
    parameter integer DATA_WIDTH = 32,
    parameter integer LATENCY    = 3,
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
    input  logic signed [DATA_WIDTH-1:0] in_data,

    output logic out_valid,
    output logic [L_W-1:0] out_l,
    output logic [D_W-1:0] out_d,
    output logic [N_W-1:0] out_n,
    output logic signed [DATA_WIDTH-1:0] out_data
	);

    generate
        if (LATENCY == 0) begin 
            always_comb begin
                out_valid = in_valid;
                out_l     = in_l;
                out_d     = in_d;
                out_n     = in_n;
                out_data  = in_data;
            end
        end
        else begin 
            logic valid_pipe [0:LATENCY-1];
            logic [L_W-1:0] l_pipe [0:LATENCY-1];
            logic [D_W-1:0] d_pipe [0:LATENCY-1];
            logic [N_W-1:0] n_pipe [0:LATENCY-1];
            logic signed [DATA_WIDTH-1:0] data_pipe [0:LATENCY-1];
            integer i;

            always_ff @(posedge clk) begin
                if (rst) begin
                    for (i = 0; i < LATENCY; i = i + 1) begin
                        valid_pipe[i] <= 1'b0;
                        l_pipe[i]     <= '0;
                        d_pipe[i]     <= '0;
                        n_pipe[i]     <= '0;
                        data_pipe[i]  <= '0;
                    end
                end
                else begin
                    valid_pipe[0] <= in_valid;
                    if (in_valid) begin
                        l_pipe[0]    <= in_l;
                        d_pipe[0]    <= in_d;
                        n_pipe[0]    <= in_n;
                        data_pipe[0] <= in_data;
                    end

                    for (i = 1; i < LATENCY; i = i + 1) begin
                        valid_pipe[i] <= valid_pipe[i-1];
                        if (valid_pipe[i-1]) begin
                            l_pipe[i]    <= l_pipe[i-1];
                            d_pipe[i]    <= d_pipe[i-1];
                            n_pipe[i]    <= n_pipe[i-1];
                            data_pipe[i] <= data_pipe[i-1];
                        end
                    end
                end
            end

            always_comb begin
                out_valid = valid_pipe[LATENCY-1];
                out_l     = l_pipe[LATENCY-1];
                out_d     = d_pipe[LATENCY-1];
                out_n     = n_pipe[LATENCY-1];
                out_data  = data_pipe[LATENCY-1];
            end
        end
    endgenerate

endmodule
