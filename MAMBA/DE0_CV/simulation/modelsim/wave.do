onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -divider {TOP LEVEL INPUTS}
add wave -noupdate /testbench/dut/clk
add wave -noupdate /testbench/dut/rst
add wave -noupdate /testbench/dut/start
add wave -noupdate -radix hexadecimal /testbench/dut/data
add wave -noupdate /testbench/dut/out_valid
add wave -noupdate -radix hexadecimal /testbench/dut/out_data
add wave -noupdate /testbench/dut/finish
add wave -noupdate -subitemconfig {{/testbench/dut/delta_A[0]} -expand} /testbench/dut/delta_A
add wave -noupdate /testbench/dut/delta_B
add wave -noupdate /testbench/dut/delta_mul_busy
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {40732624 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 150
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 1
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ns
update
WaveRestoreZoom {0 ps} {103761 ns}
