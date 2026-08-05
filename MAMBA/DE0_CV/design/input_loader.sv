module input_loader 
	#(
    parameter integer A_SIZE     = 16,
    parameter integer B_SIZE     = 16,
    parameter integer DELTA_SIZE = 16,
    parameter integer U_SIZE     = 16,
	 parameter integer C_SIZE     = 16,
	 parameter integer D_SIZE     = 16,
    parameter integer L          = 4,
    parameter integer D_IN       = 32,
    parameter integer N          = 16
	) 
	(
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic [15:0] data,

    output logic busy,
    output logic load_done,

    output logic signed [A_SIZE-1:0]     reg_A     [0:D_IN-1][0:N-1],
    output logic signed [B_SIZE-1:0]     reg_B     [0:L-1][0:N-1],
    output logic signed [DELTA_SIZE-1:0] reg_delta [0:L-1][0:D_IN-1],
    output logic signed [U_SIZE-1:0]     reg_u     [0:L-1][0:D_IN-1],
	 output logic signed [C_SIZE-1:0]	  reg_c		[0:L-1][0:N-1],
	 output logic signed [D_SIZE-1:0]	  rea_d		[0:D_IN]
	);

    typedef enum {IDLE, A, B, DELTA, U, C, D} load_state_t;

    load_state_t state;
    integer unsigned data_count;

    always_ff @(posedge clk) begin
        if (rst) begin
            state      <= IDLE;
            data_count <= 0;
            busy       <= 1'b0;
            load_done  <= 1'b0;
        end
        else begin
            load_done <= 1'b0;

            case (state)
                IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        data_count <= 0;
                        busy       <= 1'b1;
                        state      <= A;
                    end
                end

                A: begin
                    reg_A[data_count / N][data_count % N] <= $signed(data[A_SIZE-1:0]);

                    if (data_count == (D_IN*N)-1) begin
                        data_count <= 0;
                        state      <= B;
                    end
                    else begin
                        data_count <= data_count + 1;
                    end
                end

                B: begin
                    reg_B[data_count / N][data_count % N] <= $signed(data[B_SIZE-1:0]);

                    if (data_count == (L*N)-1) begin
                        data_count <= 0;
                        state      <= DELTA;
                    end
                    else begin
                        data_count <= data_count + 1;
                    end
                end

                DELTA: begin
                    reg_delta[data_count / D_IN][data_count % D_IN] <= $signed(data[DELTA_SIZE-1:0]);

                    if (data_count == (L*D_IN)-1) begin
                        data_count <= 0;
                        state      <= U;
                    end
                    else begin
                        data_count <= data_count + 1;
                    end
                end

                U: begin
                    reg_u[data_count / D_IN][data_count % D_IN] <= $signed(data[U_SIZE-1:0]);

                    if (data_count == (L*D_IN)-1) begin
                        data_count <= 0;
                        state      <= C;
                    end
                    else begin
                        data_count <= data_count + 1;
                    end
                end
					 
					 C: begin
                    reg_c[data_count / N][data_count % N] <= $signed(data[C_SIZE-1:0]);

                    if (data_count == (L*N)-1) begin
                        data_count <= 0;
                        state      <= D;
                    end
                    else begin
                        data_count <= data_count + 1;
                    end
                end
					 
					 D: begin
                    reg_d[data_count % D_IN] <= $signed(data[D_SIZE-1:0]);

                    if (data_count == D_IN-1) begin
                        data_count <= 0;
                        busy       <= 1'b0;
                        load_done  <= 1'b1;
                        state      <= IDLE;
                    end
                    else begin
                        data_count <= data_count + 1;
                    end
                end
					 
                default: begin
                    state      <= IDLE;
                    data_count <= 0;
                    busy       <= 1'b0;
                end
            endcase
        end
    end

endmodule
